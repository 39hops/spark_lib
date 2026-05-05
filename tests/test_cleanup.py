from __future__ import annotations

from typing import TYPE_CHECKING

from spark_lib import clean_columns, dedupe, drop_database_tables

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_clean_columns_normalizes_and_disambiguates(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, 2, 3)], ["ID", "Value ($)", "Value"])

    result = clean_columns(df)

    assert result.columns == ["id", "value", "value_1"]


def test_dedupe_keeps_latest_row(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [
            (1, "old", "2026-01-01"),
            (1, "new", "2026-01-02"),
            (2, "only", "2026-01-01"),
        ],
        ["id", "value", "updated_at"],
    )

    result = dedupe(df, pks="id", order_by="updated_at")
    rows = {r["id"]: r["value"] for r in result.collect()}

    assert rows == {1: "new", 2: "only"}


def test_drop_database_tables_dry_run(database: str, spark: SparkSession) -> None:
    spark.createDataFrame([(1,)], ["id"]).write.mode("overwrite").saveAsTable(
        f"{database}.table_a"
    )

    result = drop_database_tables(database, dry_run=True, max_workers=1)

    assert result == [f"{database}.table_a"]
    assert spark.catalog.tableExists(f"{database}.table_a")
