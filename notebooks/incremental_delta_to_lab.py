from typing import Any, Dict, List, Set, cast

from pyspark.sql import DataFrame, Row, Window, WindowSpec
from pyspark.sql.functions import (
    col, collect_list, desc, lower, row_number, size, struct, transform, trim,
)

from spark_lib import Input, clean_columns, quiet_azure_logging, set_spark
from spark_lib.sync import SyncSpec, SyncState, run_sync

SOURCE_ROOT: str = "abfss://<container>@<account>.dfs.core.windows.net"
SOURCE_VERSION: str = "1.2"
LAB_DB: str = "lab"
STATE_TABLE: str = f"{LAB_DB}.__spark_lib_delta_sync_state"
MAX_WORKERS: int = 8
PK_METADATA_CSV: str = "abfss://<container>@<account>.dfs.core.windows.net/config/table_primary_keys.csv"
SPARK_CONF: Dict[str, Any] = {
    "spark.databricks.delta.schema.autoMerge.enabled": "true",
}


def apply_spark_conf(conf: Dict[str, Any]) -> None:
    for key, value in conf.items():
        spark.conf.set(key, value)


apply_spark_conf(SPARK_CONF)
spark.sql(f"CREATE DATABASE IF NOT EXISTS {LAB_DB}")
set_spark(spark)
quiet_azure_logging()


def src_path_for(src_db: str, src_table: str) -> str:
    return (
        f"{SOURCE_ROOT.rstrip('/')}/"
        f"{src_db.upper()}/{src_table.upper()}/"
        f"{SOURCE_VERSION}/"
    )


def dst_table_for(src_table: str) -> str:
    return f"{LAB_DB}.{src_table}"


def load_specs(path: str) -> List[SyncSpec]:
    meta: DataFrame = clean_columns(
        Input(path, format="csv", header="true", inferSchema="true").read()
    )
    required: Set[str] = {"schema_name", "table_name", "column_name", "position"}
    missing: Set[str] = required - set(meta.columns)
    if missing:
        raise ValueError(f"PK metadata CSV is missing columns: {sorted(missing)}")

    ordered: DataFrame = (
        meta.select(
            lower(trim("schema_name")).alias("src_db"),
            lower(trim("table_name")).alias("src_table"),
            lower(trim("column_name")).alias("column_name"),
            col("position").cast("int").alias("position"),
        )
        .where(
            (col("src_db") != "")
            & (col("src_table") != "")
            & (col("column_name") != "")
            & col("position").isNotNull()
        )
        .dropDuplicates(["src_db", "src_table", "column_name", "position"])
        .withColumn("pk_struct", struct(col("position"), col("column_name")))
    )

    table_window: WindowSpec = (
        Window.partitionBy("src_db", "src_table")
        .orderBy("position")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    pick_complete: WindowSpec = Window.partitionBy("src_db", "src_table").orderBy(
        desc("pk_count"), desc("position"),
    )
    grouped: DataFrame = (
        ordered.withColumn("pk_structs", collect_list("pk_struct").over(table_window))
               .withColumn("pk_count", size("pk_structs"))
               .withColumn("__rn", row_number().over(pick_complete))
               .where(col("__rn") == 1)
               .select(
                   "src_db",
                   "src_table",
                   transform("pk_structs", lambda x: x["column_name"]).alias("pks"),
               )
    )

    specs: List[SyncSpec] = []
    for row in cast(List[Row], grouped.collect()):
        data: Dict[str, Any] = row.asDict()
        pks: List[str] = [str(c).strip() for c in data["pks"] if str(c).strip()]
        if not pks:
            raise ValueError(f"{data['src_db']}.{data['src_table']} has no primary keys in metadata CSV")
        src_db: str = str(data["src_db"]).strip()
        src_table: str = str(data["src_table"]).strip()
        specs.append({
            "src_key": f"{src_db}.{src_table}",
            "src_path": src_path_for(src_db, src_table),
            "dst_table": dst_table_for(src_table),
            "pks": pks,
        })
    return specs


specs: List[SyncSpec] = load_specs(PK_METADATA_CSV)
state: SyncState = SyncState(STATE_TABLE)
state.ensure()
successes, failures = run_sync(specs, state, max_workers=MAX_WORKERS, pool="delta_sync")

if successes:
    display(spark.createDataFrame([dict(s) for s in successes]))
if failures:
    raise RuntimeError(f"{len(failures)} table syncs failed; inspect run_parallel logs/results")
