"""
Schema registry: curates table/column descriptions for LLM context injection.

Stores table metadata (descriptions, column types, example values) and
retrieves the most relevant context for a given natural language question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ColumnMeta:
    name:        str
    type:        str                  # ClickHouse type string, e.g. "Float64"
    description: str = ""
    example:     Optional[str] = None  # Sample value as string


@dataclass
class TableMeta:
    schema:      str
    table:       str
    description: str
    columns:     List[ColumnMeta] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.table}"

    def to_context_block(self) -> str:
        """Format as a compact schema context string for LLM prompts."""
        lines = [f"Table: {self.full_name}", f"  Description: {self.description}", "  Columns:"]
        for col in self.columns:
            example = f"  (e.g. {col.example})" if col.example else ""
            lines.append(f"    - {col.name} {col.type}: {col.description}{example}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Curated table catalogue (extend as new Gold tables are added)
# ---------------------------------------------------------------------------

_CATALOGUE: List[TableMeta] = [
    TableMeta(
        schema="gold", table="sales",
        description="Daily sales transactions — one row per order line.",
        columns=[
            ColumnMeta("order_id",      "String",   "Unique order identifier",            "ORD-20240115-00042"),
            ColumnMeta("order_date",    "Date",     "Date the order was placed",           "2024-01-15"),
            ColumnMeta("product_name",  "String",   "Product SKU name",                   "Laptop Pro 15"),
            ColumnMeta("category",      "String",   "Product category",                   "Electronics"),
            ColumnMeta("region",        "String",   "Sales region",                       "APAC"),
            ColumnMeta("quantity",      "Int32",    "Units sold",                         "3"),
            ColumnMeta("unit_price",    "Float64",  "Price per unit (USD)",               "1299.99"),
            ColumnMeta("revenue",       "Float64",  "Total line revenue (qty × price)",   "3899.97"),
            ColumnMeta("_loaded_at",    "DateTime", "ETL load timestamp",                 None),
        ],
    ),
    TableMeta(
        schema="gold", table="customers",
        description="Customer master — one row per customer account.",
        columns=[
            ColumnMeta("customer_id",   "String",   "Unique customer ID",                 "CUST-00123"),
            ColumnMeta("customer_name", "String",   "Full name or company name",          "Acme Corp"),
            ColumnMeta("segment",       "String",   "Customer segment (SMB/Enterprise)",  "Enterprise"),
            ColumnMeta("country",       "String",   "Country code (ISO 3166-1 alpha-2)",  "IN"),
            ColumnMeta("created_date",  "Date",     "Account creation date",              "2022-03-10"),
            ColumnMeta("lifetime_value","Float64",  "Historical total revenue (USD)",     "45230.00"),
        ],
    ),
    TableMeta(
        schema="gold", table="orders_agg",
        description="Pre-aggregated daily order metrics — one row per (date, region, category).",
        columns=[
            ColumnMeta("agg_date",      "Date",     "Aggregation date",                   "2024-01-15"),
            ColumnMeta("region",        "String",   "Sales region",                       "EMEA"),
            ColumnMeta("category",      "String",   "Product category",                   "Software"),
            ColumnMeta("order_count",   "Int64",    "Number of distinct orders",          "142"),
            ColumnMeta("total_revenue", "Float64",  "Sum of revenue (USD)",               "58902.50"),
            ColumnMeta("avg_order_val", "Float64",  "Average order value (USD)",          "414.81"),
        ],
    ),
]

_INDEX: Dict[str, TableMeta] = {t.full_name: t for t in _CATALOGUE}


class SchemaRegistry:
    """Retrieve schema context relevant to a natural language question."""

    # Keywords → table names for simple relevance scoring
    _TABLE_HINTS: Dict[str, List[str]] = {
        "sales":        ["gold.sales", "gold.orders_agg"],
        "revenue":      ["gold.sales", "gold.orders_agg"],
        "order":        ["gold.sales", "gold.orders_agg"],
        "product":      ["gold.sales"],
        "customer":     ["gold.customers"],
        "segment":      ["gold.customers"],
        "region":       ["gold.sales", "gold.orders_agg"],
        "trend":        ["gold.orders_agg"],
        "monthly":      ["gold.orders_agg"],
        "quarter":      ["gold.orders_agg"],
        "daily":        ["gold.orders_agg"],
        "lifetime":     ["gold.customers"],
    }

    def get_context_for_question(self, question: str) -> str:
        """Return a compact schema context string for the most relevant tables."""
        relevant = self._rank_tables(question)
        if not relevant:
            relevant = _CATALOGUE  # fall back to all tables

        blocks = [t.to_context_block() for t in relevant]
        return "\n\n".join(blocks)

    def get_table(self, full_name: str) -> Optional[TableMeta]:
        return _INDEX.get(full_name)

    def list_tables(self) -> List[str]:
        return list(_INDEX.keys())

    # ------------------------------------------------------------------

    def _rank_tables(self, question: str) -> List[TableMeta]:
        lower = question.lower()
        scores: Dict[str, int] = {t.full_name: 0 for t in _CATALOGUE}
        for keyword, table_names in self._TABLE_HINTS.items():
            if keyword in lower:
                for name in table_names:
                    scores[name] = scores.get(name, 0) + 1

        # Return tables with score > 0, sorted by score descending
        ranked = sorted(
            [name for name, score in scores.items() if score > 0],
            key=lambda n: scores[n],
            reverse=True,
        )
        return [_INDEX[n] for n in ranked if n in _INDEX]
