"""Unit tests for chart spec generation and schema registry (no LLM calls)."""

import json
import pytest

from src.agent.sql_generator import _fallback_chart_spec, SQLGenerator
from src.schema.registry import SchemaRegistry


# ---------------------------------------------------------------------------
# Fallback chart spec tests (pure Python, no Bedrock)
# ---------------------------------------------------------------------------

class TestFallbackChartSpec:

    def _rows(self):
        return [
            {"product_name": "Laptop", "total_revenue": 5000.0},
            {"product_name": "Mouse",  "total_revenue": 800.0},
        ]

    def test_returns_valid_vega_lite_schema(self):
        spec = _fallback_chart_spec("Top products", ["product_name", "total_revenue"], self._rows())
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite")

    def test_bar_chart_for_nominal_x(self):
        spec = _fallback_chart_spec("Top products", ["product_name", "total_revenue"], self._rows())
        assert spec["mark"] == "bar"

    def test_line_chart_for_temporal_x(self):
        rows = [{"order_date": "2024-01-01", "revenue": 1000}]
        spec = _fallback_chart_spec("Revenue trend", ["order_date", "revenue"], rows)
        assert spec["mark"] == "line"
        assert spec["encoding"]["x"]["type"] == "temporal"

    def test_width_is_container(self):
        spec = _fallback_chart_spec("q", ["cat", "val"], [{"cat": "A", "val": 1}])
        assert spec["width"] == "container"

    def test_tooltip_includes_all_columns(self):
        cols = ["product_name", "category", "revenue"]
        rows = [{"product_name": "X", "category": "Y", "revenue": 100}]
        spec = _fallback_chart_spec("q", cols, rows)
        tooltip_fields = [t["field"] for t in spec["encoding"]["tooltip"]]
        assert set(tooltip_fields) == set(cols)

    def test_title_truncated_to_80_chars(self):
        long_q = "A" * 100
        spec = _fallback_chart_spec(long_q, ["a", "b"], [{"a": 1, "b": 2}])
        assert len(spec["title"]) <= 80


# ---------------------------------------------------------------------------
# Schema registry tests (no network calls)
# ---------------------------------------------------------------------------

class TestSchemaRegistry:

    def setup_method(self):
        self.registry = SchemaRegistry()

    def test_list_tables_returns_gold_tables(self):
        tables = self.registry.list_tables()
        assert "gold.sales" in tables
        assert "gold.customers" in tables

    def test_get_table_returns_metadata(self):
        meta = self.registry.get_table("gold.sales")
        assert meta is not None
        assert meta.full_name == "gold.sales"
        assert len(meta.columns) > 0

    def test_get_missing_table_returns_none(self):
        assert self.registry.get_table("nonexistent.table") is None

    def test_revenue_question_returns_sales_context(self):
        ctx = self.registry.get_context_for_question("Show me total revenue by region")
        assert "gold.sales" in ctx or "gold.orders_agg" in ctx

    def test_customer_question_returns_customers_context(self):
        ctx = self.registry.get_context_for_question("Which customer segment has the highest lifetime value?")
        assert "gold.customers" in ctx

    def test_context_block_includes_column_names(self):
        meta = self.registry.get_table("gold.sales")
        block = meta.to_context_block()
        assert "order_date" in block
        assert "revenue" in block

    def test_unknown_question_falls_back_to_all_tables(self):
        ctx = self.registry.get_context_for_question("xyz123 nonsense query")
        # Should still return something (all tables)
        assert len(ctx) > 0
