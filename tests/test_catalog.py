from __future__ import annotations

from typing import TYPE_CHECKING

from spark_lib.catalog import scan_databases

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_scan_databases_returns_column_dictionary(
    database: str,
    spark: SparkSession,
) -> None:
    spark.createDataFrame([(1, "a")], ["id", "value"]).write.mode(
        "overwrite"
    ).saveAsTable(f"{database}.table_a")

    result = scan_databases([database], max_workers=1)
    rows = {
        (r["table"], r["column_name"], r["data_type"])
        for r in result.select("table", "column_name", "data_type").collect()
    }

    assert ("table_a", "id", "bigint") in rows
    assert ("table_a", "value", "string") in rows
