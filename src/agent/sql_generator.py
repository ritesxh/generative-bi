"""
SQL generation and result formatting via AWS Bedrock (Claude).

SQLGenerator handles two LLM calls in the NL→SQL pipeline:
  1. generate()        — natural language question → SQL
  2. format_results()  — SQL query results → Vega-Lite chart spec + explanation
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)

_BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
_AWS_REGION    = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=_AWS_REGION)


def _invoke(prompt: str, max_tokens: int = 1024) -> str:
    """Send a prompt to Bedrock Claude and return the text response."""
    client = _bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(modelId=_BEDROCK_MODEL, body=body)
    result   = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# SQL generation prompt
# ---------------------------------------------------------------------------

_SQL_PROMPT = """\
You are a SQL expert. Given the table schema context below and a user question,
write a single read-only SQL query that answers the question.

Rules:
- Use only SELECT statements. Never use DROP, DELETE, INSERT, UPDATE, or TRUNCATE.
- Use ClickHouse SQL dialect (toStartOfMonth, toStartOfQuarter, LIMIT, etc.).
- Alias aggregated columns with descriptive names.
- Return ONLY the SQL query — no explanation, no markdown fences.

Schema context:
{schema_context}

Question: {question}

SQL:"""


# ---------------------------------------------------------------------------
# Chart spec generation prompt
# ---------------------------------------------------------------------------

_CHART_PROMPT = """\
You are a data visualisation expert. Given a SQL query and its result rows,
produce a Vega-Lite v5 chart specification (JSON only, no prose).

Choose the best chart type:
- time-series data  → line chart  (x: temporal, y: quantitative)
- category vs metric → bar chart  (x: nominal, y: quantitative)
- two metrics        → scatter    (x: quantitative, y: quantitative)
- single metric      → bar chart with one bar per row

Rules:
- Return ONLY valid JSON (Vega-Lite spec). No explanation, no markdown fences.
- Set "width": "container" and "height": 300.
- Use "$schema": "https://vega.github.io/schema/vega-lite/v5.json".
- Include a "title" derived from the question.
- Include a plain "description" field (one sentence) explaining the insight.

Question: {question}

SQL:
{sql}

Sample rows (first 5):
{sample_rows}

Column names: {columns}

Vega-Lite JSON:"""


# ---------------------------------------------------------------------------
# Explanation prompt
# ---------------------------------------------------------------------------

_EXPLAIN_PROMPT = """\
In one short paragraph (3-4 sentences), explain the following SQL query result
to a non-technical business user. Highlight the key insight clearly.

Question: {question}
SQL: {sql}
Sample rows: {sample_rows}

Explanation:"""


class SQLGenerator:
    """Handles LLM calls for SQL generation and result formatting."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL from the natural language question."""
        question       = state["question"]
        schema_context = state.get("schema_context") or "(no schema context available)"

        prompt = _SQL_PROMPT.format(schema_context=schema_context, question=question)
        try:
            sql = _invoke(prompt, max_tokens=512)
            logger.info("Generated SQL: %s", sql[:120])
        except Exception as exc:
            logger.error("SQL generation failed: %s", exc)
            sql = ""

        return {**state, "generated_sql": sql}

    def format_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Given query results, produce a Vega-Lite chart spec and explanation."""
        question = state["question"]
        sql      = state.get("generated_sql", "")
        rows: List[Dict[str, Any]] = state.get("query_results") or []

        if not rows:
            return {
                **state,
                "chart_spec": None,
                "explanation": "No results were returned for this query.",
            }

        columns     = list(rows[0].keys()) if rows else []
        sample_rows = json.dumps(rows[:5], default=str, indent=2)

        chart_spec  = self._generate_chart_spec(question, sql, sample_rows, columns)
        explanation = self._generate_explanation(question, sql, sample_rows)

        return {**state, "chart_spec": chart_spec, "explanation": explanation}

    # ------------------------------------------------------------------

    def _generate_chart_spec(
        self,
        question: str,
        sql: str,
        sample_rows: str,
        columns: List[str],
    ) -> Optional[Dict[str, Any]]:
        prompt = _CHART_PROMPT.format(
            question=question,
            sql=sql,
            sample_rows=sample_rows,
            columns=", ".join(columns),
        )
        try:
            raw = _invoke(prompt, max_tokens=1024)
            # Strip any accidental markdown fences
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            spec = json.loads(raw)
            logger.info("Chart type: %s", spec.get("mark", "unknown"))
            return spec
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Chart spec generation failed: %s", exc)
            return _fallback_chart_spec(question, columns, sample_rows)

    def _generate_explanation(self, question: str, sql: str, sample_rows: str) -> str:
        prompt = _EXPLAIN_PROMPT.format(
            question=question, sql=sql, sample_rows=sample_rows
        )
        try:
            return _invoke(prompt, max_tokens=256)
        except Exception as exc:
            logger.warning("Explanation generation failed: %s", exc)
            return "Query executed successfully."


# ---------------------------------------------------------------------------
# Fallback: rule-based Vega-Lite spec when LLM call fails
# ---------------------------------------------------------------------------

def _fallback_chart_spec(
    question: str,
    columns: List[str],
    sample_rows: str,
) -> Dict[str, Any]:
    """Generate a simple bar chart spec without an LLM call."""
    rows = json.loads(sample_rows) if isinstance(sample_rows, str) else sample_rows

    # Heuristic: first column = x (nominal), last numeric column = y
    x_field = columns[0] if columns else "category"
    y_field = columns[-1] if len(columns) > 1 else columns[0]

    # Detect if x looks temporal
    x_type = "temporal" if any(k in x_field.lower() for k in ("date", "month", "year", "time")) else "nominal"

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": question[:80],
        "description": "Auto-generated chart from query results.",
        "width": "container",
        "height": 300,
        "data": {"values": rows},
        "mark": "line" if x_type == "temporal" else "bar",
        "encoding": {
            "x": {"field": x_field, "type": x_type, "axis": {"labelAngle": -30}},
            "y": {"field": y_field, "type": "quantitative"},
            "tooltip": [{"field": col, "type": "nominal"} for col in columns],
        },
    }
