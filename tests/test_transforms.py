from __future__ import annotations

from typing import TYPE_CHECKING

from spark_lib import Input, Output, transform_df

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def test_input_output_table_round_trip(database: str, spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a")], ["id", "value"])

    Output.table(f"{database}.table_a", format="parquet").write(df)
    result = Input.table(f"{database}.table_a").read()

    assert result.collect() == df.collect()


def test_transform_df_writes_output(database: str, spark: SparkSession) -> None:
    spark.createDataFrame([(1, 10), (2, 20)], ["id", "value"]).write.mode(
        "overwrite"
    ).saveAsTable(f"{database}.input")

    @transform_df(
        output=Output.table(f"{database}.output", format="parquet"),
        source=Input.table(f"{database}.input"),
    )
    def compute(source: DataFrame) -> DataFrame:
        return source.where("value > 10")

    result = compute()

    assert result.count() == 1
    assert spark.table(f"{database}.output").first()["id"] == 2
