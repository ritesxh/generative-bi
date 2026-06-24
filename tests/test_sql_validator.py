"""Unit tests for the NL→SQL validator node."""

import pytest
from src.agent.graph import sql_validator_node


def base_state(**kwargs):
    s = {
        "session_id": "test", "question": "q", "schema_context": None,
        "generated_sql": None, "sql_valid": None, "validation_msg": None,
        "query_results": None, "chart_spec": None, "explanation": None, "errors": [],
    }
    s.update(kwargs)
    return s


class TestSQLValidator:

    def test_valid_select_passes(self):
        state = base_state(generated_sql="SELECT product, SUM(revenue) FROM gold.sales GROUP BY product")
        result = sql_validator_node(state)
        assert result["sql_valid"] is True
        assert result["validation_msg"] == "OK"

    def test_empty_sql_fails(self):
        state = base_state(generated_sql="")
        result = sql_validator_node(state)
        assert result["sql_valid"] is False

    def test_drop_table_blocked(self):
        state = base_state(generated_sql="DROP TABLE gold.sales")
        result = sql_validator_node(state)
        assert result["sql_valid"] is False
        assert "DROP" in result["validation_msg"]

    def test_delete_blocked(self):
        state = base_state(generated_sql="DELETE FROM gold.sales WHERE id = 1")
        result = sql_validator_node(state)
        assert result["sql_valid"] is False

    def test_insert_blocked(self):
        state = base_state(generated_sql="INSERT INTO gold.sales VALUES (1, 'test', 100)")
        result = sql_validator_node(state)
        assert result["sql_valid"] is False

    def test_none_sql_fails(self):
        state = base_state(generated_sql=None)
        result = sql_validator_node(state)
        assert result["sql_valid"] is False

    def test_select_with_subquery_passes(self):
        sql = """
        SELECT product_name, total_revenue
        FROM (
            SELECT product_name, SUM(revenue) AS total_revenue
            FROM gold.sales
            GROUP BY product_name
        ) ORDER BY total_revenue DESC LIMIT 10
        """
        state = base_state(generated_sql=sql)
        result = sql_validator_node(state)
        assert result["sql_valid"] is True
