"""Incremental Delta sync into managed Delta tables.

This module turns a list of source-Delta-folders into a list of
managed Delta tables in a lab database, deciding per-source whether to do
a full snapshot or an incremental Change Data Feed merge based on a tiny
state table.

Key types
---------
- :class:`SyncSpec` — TypedDict describing one source (path, dst, pks).
- :class:`SyncResult` — TypedDict for the row written to the state table.
- :class:`SyncState` — wrapper around the state Delta table with
  ``ensure``, ``get``, ``load_all``, ``upsert``, ``upsert_all``.

Key functions
-------------
- :func:`sync_delta_to_table` — sync exactly one source. Decides between
  initial snapshot, snapshot merge, no-op, or CDF merge (with snapshot
  fallback if CDF is unavailable).
- :func:`run_sync` — parallel orchestrator over many specs. Prefetches
  state in one scan, runs workers via
  :func:`spark_lib.cleanup.run_parallel`, then batches all state writes
  into one MERGE.

State table schema
------------------
The state table is created on first call to :meth:`SyncState.ensure` with
columns:

| Column | Type | Notes |
| --- | --- | --- |
| ``src_key`` | string | primary key |
| ``dst_table`` | string | |
| ``src_path`` | string | |
| ``last_delta_version`` | bigint | ``-1`` when version was unknown |
| ``sync_mode`` | string | last result mode (e.g. ``"cdf_merge"``) |
| ``synced_at`` | timestamp | set by ``upsert_all`` at write time |

Typical usage
-------------
.. code-block:: python

    from spark_lib.sync import SyncSpec, SyncState, run_sync

    specs: list[SyncSpec] = [...]
    state = SyncState("lab.__spark_lib_delta_sync_state")
    successes, failures = run_sync(specs, state, max_workers=8)
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
    """Description of one source to sync.

    Fields:
        src_key: Stable identifier, e.g. ``"sales.orders"``. Used as the
            primary key in the state table.
        src_path: Source Delta folder URI (typically ``abfss://...``).
        dst_table: Fully-qualified managed Delta table to write into,
            e.g. ``"lab.orders"``.
        pks: Primary key column names. Used for the merge predicate and for
            deduplicating CDF rows per key.
    """
    src_key: str
    src_path: str
    dst_table: str
    pks: List[str]


class SyncResult(TypedDict):
    """Outcome of one source sync.

    Fields:
        src_key: The spec's ``src_key``.
        src_path: The spec's ``src_path``.
        dst_table: The spec's ``dst_table``.
        last_delta_version: The source Delta version that was synced through.
            ``-1`` is written when the version could not be determined (the
            next run will re-snapshot rather than trust this row).
        sync_mode: One of ``"initial_snapshot"``, ``"snapshot_merge"``,
            ``"already_current"``, ``"cdf_merge"``, ``"cdf_noop"``.
    """
    src_key: str
    src_path: str
    dst_table: str
    last_delta_version: int
    sync_mode: str


@dataclass
class SyncState:
    """Persists last-synced Delta version per ``src_key``.

    Backed by a managed Delta table created on first call to :meth:`ensure`.
    The table schema is fixed (see :class:`SyncResult` for column names).

    Attributes:
        table: Fully-qualified managed Delta table name, e.g.
            ``"lab.__spark_lib_delta_sync_state"``.

    Methods at a glance:
        ensure(): Create the state table if it does not exist.
        get(src_key): Look up one row by key.
        load_all(): Read the entire state table into a dict.
        upsert(result): Apply one ``SyncResult`` row.
        upsert_all(results): Apply many ``SyncResult`` rows in one MERGE.
    """
    table: str

    def ensure(self) -> None:
        """Create the state table if it doesn't exist.

        Writes an empty Delta table with the standard sync-state schema.
        Idempotent: returns immediately if the table is already registered.
        """
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
        """Return the last synced Delta version for one source key.

        Args:
            src_key: The spec's ``src_key``.

        Returns:
            Stored ``last_delta_version`` for that key, or ``None`` if the
            state table does not exist or has no row for that key.

        Note:
            Prefer :meth:`load_all` when you're about to look up many keys —
            this method runs a small Spark query each call.
        """
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
        """Read every state row in one scan.

        Use this before parallel work to avoid N small ``get()`` queries
        against the same tiny table — one full scan beats many point reads
        when you'll need most of the rows anyway.

        Returns:
            ``{src_key: last_delta_version}`` for every row in the state
            table, or an empty dict if the table doesn't exist yet.
        """
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
        """Apply one ``SyncResult`` row to the state table.

        Convenience wrapper around :meth:`upsert_all`. Each call is a Delta
        transaction; prefer :meth:`upsert_all` when writing many results.
        """
        self.upsert_all([result])

    def upsert_all(self, results: List[SyncResult]) -> None:
        """Apply many ``SyncResult`` rows in a single Delta transaction.

        ``synced_at`` is set to ``current_timestamp()`` at write time. The
        merge predicate is on ``src_key``: matching rows are updated,
        non-matching rows are inserted.

        Args:
            results: Rows to apply. An empty list is a no-op.

        Note:
            One MERGE has fixed commit overhead, so batching many results
            into a single call is meaningfully faster than calling
            :meth:`upsert` in a loop.
        """
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

    Decision tree (in order):

    1. ``last_version is None`` or destination missing → **initial snapshot**
       (full overwrite, table created if needed).
    2. ``current_version is None`` → **snapshot merge** (we can't tell
       what's new, but a mirror still works). ``last_delta_version=-1`` is
       written so the next run will re-snapshot rather than trust this row.
    3. ``last_version >= current_version`` → **already current**, no work.
    4. Otherwise → **CDF merge** from ``last_version + 1`` through
       ``current_version``. On any exception (CDF not enabled on source,
       missing PK columns after column normalization, etc.), logs a warning
       and falls back to snapshot merge.

    Args:
        src_key: Stable identifier for the source.
        src_path: Source Delta folder URI.
        dst_table: Fully-qualified managed Delta table name.
        pks: Primary key columns.
        state: :class:`SyncState` used to record the result. ``last_version``
            is read from here when not supplied.
        last_version: Pre-resolved last synced version for this key. When
            ``None``, falls back to ``state.get(src_key)``. Pass this
            explicitly (from :meth:`SyncState.load_all`) when running many
            syncs in parallel to avoid N small queries.
        cdf_metadata_cols: CDF metadata columns to exclude from data writes.
            Defaults to :data:`spark_lib.delta.DEFAULT_CDF_METADATA`.

    Returns:
        A :class:`SyncResult` describing the outcome. Note this function
        does not write to the state table itself — the caller (or
        :func:`run_sync`) does.

    Raises:
        ValueError: PK columns missing from the source after
            ``clean_columns``.

    Example:
        >>> result = sync_delta_to_table(
        ...     src_key="sales.orders",
        ...     src_path="abfss://raw@acct.../SALES/ORDERS/1.2/",
        ...     dst_table="lab.orders",
        ...     pks=["order_id"],
        ...     state=state,
        ... )
        >>> state.upsert(result)
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
    """Run :func:`sync_delta_to_table` for many specs in parallel.

    Optimizations:

    - **One state-table scan up front** (:meth:`SyncState.load_all`); the
      per-source ``last_version`` is threaded into each worker.
    - **One batched MERGE** at the end via :meth:`SyncState.upsert_all`.
    - State writes are sequential (after workers complete) to avoid
      concurrent Delta-transaction conflicts on the state table.

    Args:
        specs: Iterable of :class:`SyncSpec` rows.
        state: State table wrapper. ``state.ensure()`` is called for you.
        max_workers: Thread-pool size for parallel sync. Default 4.
        pool: If given, set ``spark.scheduler.pool`` on each worker thread.
            Requires Spark FAIR scheduling configured at session start.
        cdf_metadata_cols: Forwarded to :func:`sync_delta_to_table`.

    Returns:
        ``(successes, failures)``. Both lists preserve spec order on the
        success side. Failures are exception objects (not re-raised here)
        so callers can inspect, count, and decide whether to fail the run.

    Example:
        >>> specs = load_specs(...)
        >>> state = SyncState("lab.__spark_lib_delta_sync_state")
        >>> successes, failures = run_sync(
        ...     specs, state, max_workers=8, pool="delta_sync",
        ... )
        >>> if failures:
        ...     raise RuntimeError(f"{len(failures)} syncs failed")
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
