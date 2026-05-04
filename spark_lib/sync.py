"""Incremental Delta sync into managed Delta tables.

`SyncState` persists per-source last-synced version in a managed Delta table.
`sync_delta_to_table` decides between snapshot and CDF for one source.
`run_sync` orchestrates many sources in parallel and writes state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from .cleanup import clean_columns, log, run_parallel
from .delta import (
    DEFAULT_CDF_METADATA,
    cdf_merge,
    current_delta_version,
    read_cdf,
    snapshot_merge,
)
from .session import get_spark
from .transforms import Input

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, Row


class SyncSpec(TypedDict, total=False):
    """One source to sync.

    `src_key` is a stable identifier (e.g., `"db.table"`). `pks` lists primary
    key columns used for merge predicates and CDF deduplication.
    """
    src_key: str
    src_path: str
    dst_table: str
    pks: List[str]


class SyncResult(TypedDict):
    src_key: str
    src_path: str
    dst_table: str
    last_delta_version: int
    sync_mode: str


@dataclass
class SyncState:
    """Persists last-synced Delta version per source key.

    Backed by a managed Delta table created on first call to `ensure()`.
    """
    table: str

    def ensure(self) -> None:
        spark = get_spark()
        if spark.catalog.tableExists(self.table):
            return
        from pyspark.sql.types import (
            LongType, StringType, StructField, StructType, TimestampType,
        )
        schema = StructType([
            StructField("src_key", StringType(), False),
            StructField("dst_table", StringType(), False),
            StructField("src_path", StringType(), False),
            StructField("last_delta_version", LongType(), False),
            StructField("sync_mode", StringType(), False),
            StructField("synced_at", TimestampType(), False),
        ])
        (
            spark.createDataFrame([], schema)
                 .write.format("delta").mode("overwrite")
                 .saveAsTable(self.table)
        )
        log.info("created sync state table %s", self.table)

    def get(self, src_key: str) -> Optional[int]:
        spark = get_spark()
        if not spark.catalog.tableExists(self.table):
            return None
        from pyspark.sql import functions as F
        row: Optional["Row"] = (
            spark.table(self.table)
                 .where(F.col("src_key") == src_key)
                 .select("last_delta_version")
                 .first()
        )
        return None if row is None else int(row["last_delta_version"])

    def load_all(self) -> Dict[str, int]:
        """Read every state row in one scan. Use before parallel work to
        avoid N small `get()` queries against the same tiny table."""
        spark = get_spark()
        if not spark.catalog.tableExists(self.table):
            return {}
        rows = (
            spark.table(self.table)
                 .select("src_key", "last_delta_version")
                 .collect()
        )
        return {r["src_key"]: int(r["last_delta_version"]) for r in rows}

    def upsert(self, result: SyncResult) -> None:
        self.upsert_all([result])

    def upsert_all(self, results: List[SyncResult]) -> None:
        """Apply many state rows in one Delta transaction. Beats a per-row
        loop on big runs because each MERGE has fixed commit overhead."""
        if not results:
            return
        from delta.tables import DeltaTable
        from pyspark.sql import functions as F

        spark = get_spark()
        rows = (
            spark.createDataFrame([dict(r) for r in results])
                 .withColumn("synced_at", F.current_timestamp())
        )
        (
            DeltaTable.forName(spark, self.table).alias("t")
            .merge(rows.alias("s"), "t.src_key = s.src_key")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )


def sync_delta_to_table(
    *,
    src_key: str,
    src_path: str,
    dst_table: str,
    pks: Iterable[str],
    state: SyncState,
    last_version: Optional[int] = None,
    cdf_metadata_cols: Set[str] = DEFAULT_CDF_METADATA,
) -> SyncResult:
    """Sync one Delta source folder into one managed Delta table.

    First-time runs and any source where the version can't be determined fall
    through to a snapshot path: an initial overwrite when the destination
    doesn't exist, otherwise a snapshot MERGE that mirrors deletes. When the
    destination exists and CDF reads succeed, applies an incremental CDF
    merge instead. CDF failures fall back to snapshot.
    """
    spark = get_spark()
    pk_list: List[str] = list(pks)
    current_version: Optional[int] = current_delta_version(src_path)
    if last_version is None:
        last_version = state.get(src_key)
    snapshot_version: int = (
        current_version if current_version is not None else -1
    )

    def _snapshot(version: int) -> SyncResult:
        df = clean_columns(Input(src_path, format="delta").read())
        missing: List[str] = [c for c in pk_list if c not in df.columns]
        if missing:
            raise ValueError(
                f"{dst_table} missing PK columns after clean_columns: {missing}"
            )
        if not spark.catalog.tableExists(dst_table):
            log.info("%s -> %s initial snapshot", src_key, dst_table)
            (
                df.write.format("delta").mode("overwrite")
                  .saveAsTable(dst_table)
            )
            return _result(src_key, src_path, dst_table, version, "initial_snapshot")
        log.info("%s -> %s snapshot merge", src_key, dst_table)
        snapshot_merge(dst_table, df, pk_list, delete_unmatched=True)
        return _result(src_key, src_path, dst_table, version, "snapshot_merge")

    if last_version is None or not spark.catalog.tableExists(dst_table):
        return _snapshot(snapshot_version)
    if current_version is None:
        log.warning(
            "%s: no readable Delta version at %s; doing snapshot merge",
            src_key, src_path,
        )
        return _snapshot(snapshot_version)
    if last_version >= current_version:
        return _result(src_key, src_path, dst_table, current_version, "already_current")

    try:
        log.info(
            "%s -> %s CDF merge from %d to %d",
            src_key, dst_table, last_version + 1, current_version,
        )
        cdf = clean_columns(read_cdf(src_path, last_version + 1))
        missing = [c for c in pk_list if c not in cdf.columns]
        if missing:
            raise ValueError(
                f"{dst_table} CDF missing PK columns after clean_columns: {missing}"
            )
        applied: bool = cdf_merge(
            dst_table, cdf, pk_list, metadata_cols=cdf_metadata_cols,
        )
        return _result(
            src_key, src_path, dst_table, current_version,
            "cdf_merge" if applied else "cdf_noop",
        )
    except Exception as exc:
        log.warning(
            "%s: CDF unavailable (%s), falling back to snapshot merge",
            src_key, exc,
        )
        return _snapshot(current_version)


def run_sync(
    specs: Iterable[SyncSpec],
    state: SyncState,
    *,
    max_workers: int = 4,
    pool: Optional[str] = None,
    cdf_metadata_cols: Set[str] = DEFAULT_CDF_METADATA,
) -> Tuple[List[SyncResult], List[BaseException]]:
    """Run `sync_delta_to_table` for many specs in parallel and write state.

    State writes happen serially after workers return to avoid concurrent
    Delta transaction conflicts on the state table.
    """
    state.ensure()
    # One scan of the state table up-front; workers no longer hit it.
    last_versions: Dict[str, int] = state.load_all()
    spec_list: List[SyncSpec] = list(specs)
    jobs: List[Dict[str, Any]] = [
        {
            "name": s["src_key"],
            "src_key": s["src_key"],
            "src_path": s["src_path"],
            "dst_table": s["dst_table"],
            "pks": list(s["pks"]),
            "last_version": last_versions.get(s["src_key"]),
        }
        for s in spec_list
    ]

    def _run(
        src_key: str,
        src_path: str,
        dst_table: str,
        pks: List[str],
        last_version: Optional[int],
        name: str = "",
    ) -> SyncResult:
        return sync_delta_to_table(
            src_key=src_key,
            src_path=src_path,
            dst_table=dst_table,
            pks=pks,
            state=state,
            last_version=last_version,
            cdf_metadata_cols=cdf_metadata_cols,
        )

    raw: List[Union[SyncResult, BaseException]] = run_parallel(
        _run, jobs, max_workers=max_workers, pool=pool, fail_fast=False,
    )
    successes: List[SyncResult] = [
        r for r in raw if not isinstance(r, BaseException)
    ]
    failures: List[BaseException] = [
        r for r in raw if isinstance(r, BaseException)
    ]
    # One MERGE for all state writes instead of one transaction per success.
    state.upsert_all(successes)
    log.info(
        "run_sync: %d succeeded, %d failed", len(successes), len(failures),
    )
    return successes, failures


def _result(
    src_key: str,
    src_path: str,
    dst_table: str,
    version: int,
    mode: str,
) -> SyncResult:
    return {
        "src_key": src_key,
        "src_path": src_path,
        "dst_table": dst_table,
        "last_delta_version": int(version),
        "sync_mode": mode,
    }


__all__: List[str] = [
    "SyncResult",
    "SyncSpec",
    "SyncState",
    "run_sync",
    "sync_delta_to_table",
]
