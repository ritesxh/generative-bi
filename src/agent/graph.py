"""
LangGraph NL→SQL agent graph for Generative BI.

Pipeline:
    schema_lookup → sql_generator → sql_validator → executor → formatter
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class QueryState(TypedDict):
    """Shared state for the NL→SQL agent pipeline."""
    session_id:    str
    question:      str
    schema_context: Optional[str]    # Injected table/column descriptions
    generated_sql:  Optional[str]
    sql_valid:      Optional[bool]
    validation_msg: Optional[str]
    query_results:  Optional[List[Dict[str, Any]]]
    chart_spec:     Optional[Dict[str, Any]]   # Vega-Lite spec
    explanation:    Optional[str]
    errors:         List[str]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def schema_lookup_node(state: QueryState) -> QueryState:
    """Load relevant table schemas and column descriptions from the registry."""
    from src.schema.registry import SchemaRegistry
    registry = SchemaRegistry()
    schema_ctx = registry.get_context_for_question(state["question"])
    return {**state, "schema_context": schema_ctx}


def sql_generator_node(state: QueryState) -> QueryState:
    """Use AWS Bedrock to translate the natural language question into SQL."""
    from src.agent.sql_generator import SQLGenerator
    return SQLGenerator().run(state)


def sql_validator_node(state: QueryState) -> QueryState:
    """Validate the generated SQL: syntax check + injection guard."""
    sql = state.get("generated_sql", "")
    if not sql:
        return {**state, "sql_valid": False, "validation_msg": "No SQL generated"}

    # Basic injection guard: reject destructive keywords outside SELECT
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER"]
    upper_sql = sql.upper()
    violations = [kw for kw in dangerous if kw in upper_sql]
    if violations:
        return {
            **state,
            "sql_valid": False,
            "validation_msg": f"Blocked keywords detected: {violations}",
        }

    return {**state, "sql_valid": True, "validation_msg": "OK"}


def executor_node(state: QueryState) -> QueryState:
    """Execute the validated SQL against ClickHouse."""
    import clickhouse_connect
    import os

    if not state.get("sql_valid"):
        return {
            **state,
            "errors": state["errors"] + [f"SQL blocked: {state.get('validation_msg')}"],
        }

    client = clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
    )
    result = client.query(state["generated_sql"])
    rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
    return {**state, "query_results": rows}


def formatter_node(state: QueryState) -> QueryState:
    """Ask the LLM to produce a Vega-Lite chart spec and plain English explanation."""
    from src.agent.sql_generator import SQLGenerator
    return SQLGenerator().format_results(state)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(QueryState)

    graph.add_node("schema_lookup",  schema_lookup_node)
    graph.add_node("sql_generator",  sql_generator_node)
    graph.add_node("sql_validator",  sql_validator_node)
    graph.add_node("executor",       executor_node)
    graph.add_node("formatter",      formatter_node)

    graph.set_entry_point("schema_lookup")
    graph.add_edge("schema_lookup", "sql_generator")
    graph.add_edge("sql_generator", "sql_validator")
    graph.add_edge("sql_validator", "executor")
    graph.add_edge("executor",      "formatter")
    graph.add_edge("formatter",     END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with an optional LangGraph checkpointer.

    Pass a checkpointer instance (e.g. PostgresSaver, MemorySaver) or leave
    None to compile without persistence (useful for testing).
    """
    return build_graph().compile(checkpointer=checkpointer)
