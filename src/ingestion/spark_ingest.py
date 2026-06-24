"""
Apache Spark ingestion pipeline: raw data → Gold (ClickHouse).

Reads raw CSV/JSON/Parquet files, applies cleansing and typing,
writes to ClickHouse Gold table via JDBC connector.

Usage:
    spark-submit src/ingestion/spark_ingest.py \
        --source s3a://raw-bucket/sales/ \
        --table gold.sales \
        --format parquet
"""

import argparse
import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, TimestampType

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.environ.get("CLICKHOUSE_PASSWORD", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("generative-bi-ingest")
        .config("spark.jars.packages", "com.clickhouse:clickhouse-jdbc:0.6.0")
        .getOrCreate()
    )


def read_source(spark: SparkSession, path: str, fmt: str) -> DataFrame:
    logger.info("Reading %s from %s", fmt, path)
    reader = spark.read.option("header", "true").option("inferSchema", "true")
    return getattr(reader, fmt)(path)


def cleanse(df: DataFrame) -> DataFrame:
    """Drop null key columns and add load metadata."""
    now_ts = F.lit(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")).cast(TimestampType())
    return (
        df.dropDuplicates()
        .na.drop(how="all")
        .withColumn("_loaded_at", now_ts)
        .withColumn("_source", F.input_file_name())
    )


def write_clickhouse(df: DataFrame, table: str) -> None:
    """Write Spark DataFrame to ClickHouse via JDBC."""
    jdbc_url = f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}"
    logger.info("Writing %d rows to ClickHouse table: %s", df.count(), table)
    (
        df.write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("user", CLICKHOUSE_USER)
        .option("password", CLICKHOUSE_PASS)
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
        .mode("append")
        .save()
    )
    logger.info("Write complete: %s", table)


def main():
    parser = argparse.ArgumentParser(description="Spark → ClickHouse Gold ingest")
    parser.add_argument("--source", required=True, help="Input path (S3, local, or HDFS)")
    parser.add_argument("--table",  required=True, help="Target ClickHouse table (schema.table)")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv", "json"])
    args = parser.parse_args()

    spark  = build_spark()
    raw_df = read_source(spark, args.source, args.format)
    clean  = cleanse(raw_df)
    write_clickhouse(clean, args.table)


if __name__ == "__main__":
    main()
