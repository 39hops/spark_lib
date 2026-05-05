from __future__ import annotations

from typing import TYPE_CHECKING

from spark_lib.delta import column_map, merge_condition, snapshot_merge

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_merge_condition_and_column_map() -> None:
    assert merge_condition(["id", "fk_id"]) == "t.id = s.id AND t.fk_id = s.fk_id"
    assert column_map(["id", "value"]) == {"id": "s.id", "value": "s.value"}


def test_snapshot_merge_updates_inserts_and_deletes(
    database: str,
    spark: SparkSession,
) -> None:
    spark.createDataFrame([(1, "old"), (2, "delete")], ["id", "value"]).write.format(
        "delta"
    ).mode("overwrite").saveAsTable(f"{database}.target")
    source = spark.createDataFrame([(1, "new"), (3, "insert")], ["id", "value"])

    snapshot_merge(f"{database}.target", source, on=["id"])
    rows = {
        r["id"]: r["value"]
        for r in spark.table(f"{database}.target").orderBy("id").collect()
    }

    assert rows == {1: "new", 3: "insert"}
