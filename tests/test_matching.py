from __future__ import annotations

from typing import TYPE_CHECKING

from spark_lib.matching import infer_key_from_text, normalize_text

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def test_normalize_text(spark: SparkSession) -> None:
    df = spark.createDataFrame([(" A-B  C! ",)], ["value"])

    result = df.select(normalize_text("value").alias("value")).first()

    assert result["value"] == "a b c"


def test_infer_key_from_text_exact_first(spark: SparkSession) -> None:
    from pyspark.sql.types import LongType, StringType, StructField, StructType

    left_schema = StructType([
        StructField("id", LongType(), True),
        StructField("value", StringType(), True),
    ])
    left = spark.createDataFrame([(None, "A B C")], left_schema)
    reference = spark.createDataFrame([(7, "abc")], ["id", "value_ref"])

    result = infer_key_from_text(
        left,
        reference,
        key_col="id",
        text_col="value",
        reference_text_col="value_ref",
        method="ml",
    )
    row = result.first()

    assert row["id"] == 7
    assert row["_match_inferred_id"] == 7
    assert row["_match_score"] == 1.0
    assert row["_match_distance"] == 0.0
