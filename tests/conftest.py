from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def spark(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SparkSession]:
    pyspark = pytest.importorskip("pyspark.sql")
    delta = pytest.importorskip("delta")

    from spark_lib import set_spark

    warehouse = tmp_path_factory.mktemp("warehouse")
    builder = (
        pyspark.SparkSession.builder
        .master("local[2]")
        .appName("spark-lib-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    try:
        session = delta.configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception as exc:
        pytest.skip(f"local Spark session unavailable: {exc}")

    set_spark(session)
    yield session
    session.stop()


@pytest.fixture()
def database(spark: SparkSession) -> Iterator[str]:
    name = "test_db"
    spark.sql(f"DROP DATABASE IF EXISTS {name} CASCADE")
    spark.sql(f"CREATE DATABASE {name}")
    yield name
    spark.sql(f"DROP DATABASE IF EXISTS {name} CASCADE")
