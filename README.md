# Generative BI — AI Data Platform

> A full-stack AI data platform enabling **natural language querying** over curated ClickHouse datasets via AWS Bedrock. A LangChain + LangGraph agent translates plain English questions into SQL, executes them against ClickHouse, and returns chart-ready data products — with Apache Spark driving the ingestion pipelines.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                │
│  REST APIs  |  CSV uploads  |  Kafka topics  |  S3 files          │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│             APACHE SPARK INGESTION PIPELINES                       │
│  Raw → Bronze (S3/Parquet)  →  Silver (cleansed)  →              │
│  Gold (ClickHouse — curated data products)                        │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│                    CLICKHOUSE (Curated Layer)                       │
│  Columnar, sub-second analytics queries                            │
│  Schema registered in PostgreSQL (with lineage + descriptions)    │
└─────────────────────────┬─────────────────────────────────────────┘
                          │  schema context injected
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│              LANGGRAPH NL→SQL AGENT (AWS Bedrock)                  │
│                                                                     │
│  [schema_lookup] → [sql_generator] → [sql_validator]              │
│        → [executor] → [formatter] → [chart_builder]               │
│                                                                     │
│  LangGraph checkpointing in PostgreSQL (multi-turn memory)         │
│  JWT / OAuth2 bearer token authentication                          │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│                    FASTAPI  +  REACT FRONTEND                      │
│  Natural language input → chart + table + SQL explanation          │
│  Dashboard builder, saved queries, usage analytics                 │
└───────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---|---|
| **NL→SQL agent** | LangChain + LangGraph agent with schema context injection |
| **Multi-turn memory** | LangGraph checkpointing in PostgreSQL — remembers prior questions |
| **Spark ingestion** | Parquet-backed pipelines write curated Gold data to ClickHouse |
| **Schema curation** | Schema registry with column descriptions, lineage, example values |
| **JWT auth** | OAuth2 / JWT bearer token authentication |
| **Chart delivery** | Agent returns Vega-Lite chart spec for React frontend to render |

---

## Technology Stack

| Layer | Technology |
|---|---|
| NL→SQL agent | LangChain, LangGraph, AWS Bedrock (Claude) |
| Data ingestion | Apache Spark (PySpark), Parquet |
| Analytics DB | ClickHouse |
| State / schema | PostgreSQL |
| API | FastAPI |
| Auth | OAuth2 + JWT |
| Frontend | React |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Quick Start

```bash
git clone https://github.com/ritesxh/generative-bi.git
cd generative-bi

cp .env.example .env
# Set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, CLICKHOUSE_HOST, POSTGRES_URL

docker compose up -d

# Ingest sample data
python src/ingestion/spark_ingest.py --source sample_data/sales.csv --table gold.sales

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me monthly revenue trend for last 6 months"}'
```

---

## Project Structure

```
generative-bi/
├── src/
│   ├── agent/
│   │   ├── graph.py           # LangGraph agent graph (NL→SQL pipeline)
│   │   ├── sql_generator.py   # LLM-powered SQL generation
│   │   └── schema_loader.py   # ClickHouse schema introspection
│   ├── ingestion/
│   │   └── spark_ingest.py    # Spark: raw → Gold (ClickHouse)
│   ├── api/
│   │   └── main.py            # FastAPI: /query, /schema, /health
│   └── schema/
│       └── registry.py        # Schema curation + descriptions
├── tests/
├── docker-compose.yml
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## Example Query

**Input:** *"What were the top 5 products by revenue last quarter?"*

**Generated SQL:**
```sql
SELECT
    product_name,
    SUM(revenue) AS total_revenue
FROM gold.sales
WHERE order_date >= toStartOfQuarter(now() - INTERVAL 1 QUARTER)
  AND order_date <  toStartOfQuarter(now())
GROUP BY product_name
ORDER BY total_revenue DESC
LIMIT 5
```

**Output:** Vega-Lite bar chart spec + tabular data + SQL explanation.

---

## License

MIT
