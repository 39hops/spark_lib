# Notebook source
# Incremental Delta Sync To Lab Managed Tables
# Generated from notebooks/incremental_delta_to_lab.ipynb.

# COMMAND ----------
# # Incremental Delta Sync To Lab Managed Tables
# 
# Reads source Delta tables from `{src_db}/{src_table}/{version}/` in ADLS and writes to managed Delta tables in a lab database.
# 
# Use Delta Change Data Feed when available. If CDF is not enabled on a source table, fall back to a full snapshot `MERGE` with `whenNotMatchedBySourceDelete()` so removed source rows are removed from the lab table too.
# 
# Do not parse `_delta_log`, checkpoints, or deletion vectors yourself. `spark.read.format("delta")` and Delta Lake already apply those correctly.

# COMMAND ----------
# Configure these values for your environment.
SOURCE_ROOT: str = "abfss://<container>@<account>.dfs.core.windows.net"
SOURCE_VERSION: str = "1.2"
LAB_DB: str = "lab"
STATE_TABLE: str = f"{LAB_DB}.__spark_lib_delta_sync_state"
MAX_WORKERS: int = 8

# Metadata CSV read through spark_lib.Input. Expected columns:
#   schema_name, table_name, column_name, position
# Each row represents one PK column. Composite keys are ordered by `position`.
PK_METADATA_CSV: str = "abfss://<container>@<account>.dfs.core.windows.net/config/table_primary_keys.csv"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {LAB_DB}")
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

# COMMAND ----------
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, TypedDict, Union, cast

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Row, Window, WindowSpec
from pyspark.sql.functions import (
    col,
    collect_list,
    current_timestamp,
    desc,
    lower,
    row_number,
    size,
    struct,
    transform,
    trim,
)
from pyspark.sql.types import LongType, StringType, StructField, StructType, TimestampType

from spark_lib import Input, Output, clean_columns, quiet_azure_logging, run_parallel, set_spark

set_spark(spark)
quiet_azure_logging()

logger: logging.Logger = logging.getLogger("spark_lib.delta_sync")
if not logger.handlers:
    handler: logging.Handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class TableSpec(TypedDict, total=False):
    src_db: str
    src_table: str
    pks: List[str]
    dst_table: str


class SyncResult(TypedDict):
    src_db: str
    src_table: str
    dst_table: str
    src_path: str
    last_delta_version: int
    sync_mode: str


CDF_METADATA_COLUMNS: Set[str] = {"change_type", "commit_version", "commit_timestamp"}


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_table_specs(path: str) -> List[TableSpec]:
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
        .withColumn(
            "pk_struct",
            struct(col("position"), col("column_name")),
        )
    )

    table_window: WindowSpec = (
        Window.partitionBy("src_db", "src_table")
        .orderBy("position")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    complete_record_window: WindowSpec = Window.partitionBy("src_db", "src_table").orderBy(
        desc("pk_count"),
        desc("position"),
    )
    grouped: DataFrame = (
        ordered.withColumn("pk_structs", collect_list("pk_struct").over(table_window))
        .withColumn("pk_count", size("pk_structs"))
        .withColumn("__rn", row_number().over(complete_record_window))
        .where(col("__rn") == 1)
        .select(
            "src_db",
            "src_table",
            transform("pk_structs", lambda x: x["column_name"]).alias("pks"),
        )
    )

    specs: List[TableSpec] = []
    for row in cast(List[Row], grouped.collect()):
        data: Dict[str, Any] = row.asDict()
        pks: List[str] = [clean_str(col) for col in data["pks"] if clean_str(col)]
        if not pks:
            raise ValueError(f"{data['src_db']}.{data['src_table']} has no primary keys in metadata CSV")
        specs.append(
            {
                "src_db": clean_str(data["src_db"]),
                "src_table": clean_str(data["src_table"]),
                "pks": pks,
            }
        )
    return specs

TABLES: List[TableSpec] = load_table_specs(PK_METADATA_CSV)
logger.info("loaded %d table specs from %s", len(TABLES), PK_METADATA_CSV)


def src_path_for(spec: TableSpec) -> str:
    return f"{SOURCE_ROOT.rstrip('/')}/{spec['src_db']}/{spec['src_table']}/{SOURCE_VERSION}/"


def dst_table_for(spec: TableSpec) -> str:
    name: Optional[str] = spec.get("dst_table")
    if name:
        return name if "." in name else f"{LAB_DB}.{name}"
    return f"{LAB_DB}.{spec['src_db']}__{spec['src_table']}"


def require_pks(spec: TableSpec) -> List[str]:
    pks: List[str] = list(spec.get("pks") or [])
    if not pks:
        raise ValueError(f"{spec['src_db']}.{spec['src_table']} needs pks for merge/delete sync")
    return pks


def table_exists(name: str) -> bool:
    return bool(spark.catalog.tableExists(name))


def current_delta_version(path: str) -> int:
    row: Optional[Row] = DeltaTable.forPath(spark, path).history(1).select("version").first()
    if row is None:
        raise ValueError(f"No Delta history found at {path}")
    return int(row["version"])


def ensure_state_table() -> None:
    if table_exists(STATE_TABLE):
        return
    schema: StructType = StructType([
        StructField("src_db", StringType(), False),
        StructField("src_table", StringType(), False),
        StructField("dst_table", StringType(), False),
        StructField("src_path", StringType(), False),
        StructField("last_delta_version", LongType(), False),
        StructField("sync_mode", StringType(), False),
        StructField("synced_at", TimestampType(), False),
    ])
    spark.createDataFrame([], schema).write.format("delta").mode("overwrite").saveAsTable(STATE_TABLE)
    logger.info("created sync state table %s", STATE_TABLE)


def last_synced_version(spec: TableSpec) -> Optional[int]:
    if not table_exists(STATE_TABLE):
        return None
    row: Optional[Row] = (
        spark.table(STATE_TABLE)
             .where((col("src_db") == spec["src_db"]) & (col("src_table") == spec["src_table"]))
             .select("last_delta_version")
             .first()
    )
    return None if row is None else int(row["last_delta_version"])


def upsert_state(result: SyncResult) -> None:
    row: DataFrame = spark.createDataFrame([dict(result)]).withColumn("synced_at", current_timestamp())
    DeltaTable.forName(spark, STATE_TABLE).alias("t").merge(
        row.alias("s"),
        "t.src_db = s.src_db AND t.src_table = s.src_table",
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()


def merge_condition(pks: Iterable[str]) -> str:
    return " AND ".join(f"t.{col} = s.{col}" for col in pks)


def column_map(cols: Iterable[str], prefix: str = "s") -> Dict[str, str]:
    return {col: f"{prefix}.{col}" for col in cols}


def sync_result(
    spec: TableSpec,
    dst_table: str,
    src_path: str,
    version: int,
    mode: str,
) -> SyncResult:
    return {
        "src_db": spec["src_db"],
        "src_table": spec["src_table"],
        "dst_table": dst_table,
        "src_path": src_path,
        "last_delta_version": int(version),
        "sync_mode": mode,
    }


ensure_state_table()

# COMMAND ----------
def full_snapshot_merge(spec: TableSpec, src_path: str, dst_table: str, version: int) -> SyncResult:
    pks: List[str] = require_pks(spec)
    df: DataFrame = clean_columns(Input(src_path, format="delta").read())

    missing: List[str] = [col for col in pks if col not in df.columns]
    if missing:
        raise ValueError(f"{dst_table} missing PK columns after clean_columns: {missing}")

    if not table_exists(dst_table):
        logger.info("%s.%s -> %s initial snapshot", spec["src_db"], spec["src_table"], dst_table)
        Output.table(dst_table, mode="overwrite", format="delta").write(df)
        return sync_result(spec, dst_table, src_path, version, "initial_snapshot")

    logger.info("%s.%s -> %s snapshot merge", spec["src_db"], spec["src_table"], dst_table)
    cols: List[str] = df.columns
    (
        DeltaTable.forName(spark, dst_table).alias("t")
        .merge(df.alias("s"), merge_condition(pks))
        .whenMatchedUpdate(set=column_map(cols))
        .whenNotMatchedInsert(values=column_map(cols))
        .whenNotMatchedBySourceDelete()
        .execute()
    )
    return sync_result(spec, dst_table, src_path, version, "snapshot_merge")


def cdf_merge(
    spec: TableSpec,
    src_path: str,
    dst_table: str,
    start_version: int,
    current_version: int,
) -> SyncResult:
    logger.info("%s.%s -> %s CDF merge from version %d to %d", spec["src_db"], spec["src_table"], dst_table, start_version, current_version)
    pks: List[str] = require_pks(spec)
    cdf: DataFrame = (
        spark.read.format("delta")
             .option("readChangeFeed", "true")
             .option("startingVersion", int(start_version))
             .load(src_path)
    )
    cdf = clean_columns(cdf).where(col("change_type") != "update_preimage")

    if cdf.limit(1).count() == 0:
        return sync_result(spec, dst_table, src_path, current_version, "cdf_noop")

    missing: List[str] = [col for col in pks if col not in cdf.columns]
    if missing:
        raise ValueError(f"{dst_table} CDF missing PK columns after clean_columns: {missing}")

    window: WindowSpec = Window.partitionBy(*pks).orderBy(
        desc("commit_version"),
        desc("commit_timestamp"),
    )
    latest: DataFrame = (
        cdf.withColumn("__rn", row_number().over(window))
           .where(col("__rn") == 1)
           .drop("__rn")
    )
    data_cols: List[str] = [col for col in latest.columns if col not in CDF_METADATA_COLUMNS]

    (
        DeltaTable.forName(spark, dst_table).alias("t")
        .merge(latest.alias("s"), merge_condition(pks))
        .whenMatchedDelete(condition="s.change_type = 'delete'")
        .whenMatchedUpdate(condition="s.change_type <> 'delete'", set=column_map(data_cols))
        .whenNotMatchedInsert(condition="s.change_type <> 'delete'", values=column_map(data_cols))
        .execute()
    )
    return sync_result(spec, dst_table, src_path, current_version, "cdf_merge")


def sync_table(spec: TableSpec) -> SyncResult:
    src_path: str = src_path_for(spec)
    dst_table: str = dst_table_for(spec)
    current_version: int = current_delta_version(src_path)
    last_version: Optional[int] = last_synced_version(spec)

    if last_version is None or not table_exists(dst_table):
        return full_snapshot_merge(spec, src_path, dst_table, current_version)
    if last_version >= current_version:
        return sync_result(spec, dst_table, src_path, current_version, "already_current")

    try:
        return cdf_merge(spec, src_path, dst_table, last_version + 1, current_version)
    except Exception as exc:
        # Common reason: source table was not created with delta.enableChangeDataFeed=true.
        logger.warning("CDF failed for %s.%s; falling back to snapshot merge: %s", spec["src_db"], spec["src_table"], exc)
        return full_snapshot_merge(spec, src_path, dst_table, current_version)

# COMMAND ----------
jobs: List[Dict[str, TableSpec]] = [{"spec": spec} for spec in TABLES]
results: List[Union[SyncResult, BaseException]] = run_parallel(
    sync_table,
    jobs,
    max_workers=MAX_WORKERS,
    pool="delta_sync",
    fail_fast=False,
)

failures: List[BaseException] = [r for r in results if isinstance(r, BaseException)]
successes: List[SyncResult] = [cast(SyncResult, r) for r in results if not isinstance(r, BaseException)]

# State writes are sequential to avoid concurrent Delta transaction conflicts on the state table.
for result in successes:
    upsert_state(result)

logger.info("sync completed: %d succeeded, %d failed", len(successes), len(failures))

if successes:
    display(spark.createDataFrame([dict(r) for r in successes]))
if failures:
    raise RuntimeError(f"{len(failures)} table syncs failed; inspect run_parallel logs/results")
