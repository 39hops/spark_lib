"""Delta Lake primitives: version lookup, snapshot merge, CDF merge.

This module wraps the small set of `DeltaTable` boilerplate used across
ingestion notebooks. It intentionally contains no project-specific concerns
(no path conventions, no state tables, no naming logic) so the helpers
compose cleanly with whatever orchestration sits on top.

Public API
----------
- ``current_delta_version(path)`` — file-only Delta version lookup.
- ``snapshot_merge(target, df, on, *, delete_unmatched=True)`` — full upsert
  with optional source-driven deletes.
- ``cdf_merge(target, cdf, on)`` — apply a Change Data Feed onto a target.
- ``read_cdf(path, start_version)`` — small reader convenience.
- ``merge_condition(pks)`` and ``column_map(cols)`` — SQL-string builders.
- ``DEFAULT_CDF_METADATA`` — CDF metadata column names.

Conventions
-----------
- All functions assume Synapse-style paths (``abfss://``) and a registered
  or active SparkSession (see :mod:`spark_lib.session`).
- The target of a merge must already exist as a managed Delta table; the
  higher-level :mod:`spark_lib.sync` module handles the create-if-missing
  case.
- ``current_delta_version`` uses ``notebookutils``/``mssparkutils`` for
  filesystem reads. Outside Synapse it returns ``None``.
"""
from __future__ import annotations

import builtins
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

from .session import get_spark

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


DEFAULT_CDF_METADATA: Set[str] = {
    "change_type",
    "commit_version",
    "commit_timestamp",
}


def merge_condition(
    pks: Iterable[str],
    target_alias: str = "t",
    source_alias: str = "s",
) -> str:
    """Build a SQL merge predicate joining target and source on ``pks``.

    Args:
        pks: Primary key columns to equate between target and source.
        target_alias: Alias for the target side of the merge. Default ``"t"``.
        source_alias: Alias for the source side of the merge. Default ``"s"``.

    Returns:
        A single SQL string of the form ``"t.k1 = s.k1 AND t.k2 = s.k2"``.

    Example:
        >>> merge_condition(["id", "fk_id"])
        't.id = s.id AND t.fk_id = s.fk_id'
    """
    return " AND ".join(
        f"{target_alias}.{c} = {source_alias}.{c}" for c in pks
    )


def column_map(cols: Iterable[str], prefix: str = "s") -> Dict[str, str]:
    """Build the ``{column: "<prefix>.<column>"}`` map used by Delta merges.

    Used as ``set=`` for ``whenMatchedUpdate`` and ``values=`` for
    ``whenNotMatchedInsert`` so the merge writes every column from the
    source side.

    Args:
        cols: Columns to include.
        prefix: Source alias used in the SQL expression. Default ``"s"``.

    Returns:
        A dict mapping each column name to a SQL expression string.

    Example:
        >>> column_map(["id", "value"])
        {'id': 's.id', 'value': 's.value'}
    """
    return {c: f"{prefix}.{c}" for c in cols}


def current_delta_version(path: str) -> Optional[int]:
    """Cheap file-only lookup of the latest Delta commit at ``path``.

    Reads ``_delta_log/_last_checkpoint`` first (a tiny JSON file naming the
    latest checkpoint version), falling back to listing the log directory
    and picking the max numeric ``.json`` commit name. Returns ``None`` when
    neither is readable so callers can decide whether to keep going.

    Why not ``DeltaTable.forPath`` or ``DESCRIBE HISTORY``:

    - ``forPath`` is URI-strict and rejects some otherwise-readable Delta
      folders (Synapse Link landing zones, trailing-slash quirks).
    - ``DESCRIBE HISTORY`` triggers a full log scan even when only the
      latest version is needed.

    Args:
        path: Source Delta folder URI (typically ``abfss://...``).

    Returns:
        The latest committed version, or ``None`` if it cannot be determined
        (folder unreadable, no ``_delta_log``, or notebookutils unavailable).

    Example:
        >>> v = current_delta_version("abfss://container@acct.../path/to/table/")
        >>> if v is None:
        ...     # treat as snapshot
        ...     ...
    """
    nb = _nbutils()
    if nb is None:
        return None
    p: str = path.rstrip("/")
    log_dir: str = f"{p}/_delta_log"
    try:
        cp_text: str = nb.fs.head(f"{log_dir}/_last_checkpoint", 4096)
        version: Any = json.loads(cp_text).get("version")
        if isinstance(version, int):
            return version
    except Exception:
        pass
    try:
        entries: List[Any] = list(nb.fs.ls(log_dir))
    except Exception:
        return None
    versions: List[int] = [
        int(e.name.split(".")[0])
        for e in entries
        if e.name.endswith(".json") and e.name[0].isdigit()
    ]
    return builtins.max(versions) if versions else None


def snapshot_merge(
    target_table: str,
    df: "DataFrame",
    on: Iterable[str],
    *,
    delete_unmatched: bool = True,
) -> None:
    """Upsert ``df`` into ``target_table`` with optional source-driven deletes.

    Wraps the ``forName + merge + whenMatchedUpdate(set=...) +
    whenNotMatchedInsert(values=...) + (whenNotMatchedBySourceDelete)`` chain
    into one call. The target must already exist as a managed Delta table.

    Args:
        target_table: Fully-qualified managed Delta table name
            (e.g. ``"db.table"``).
        df: Source DataFrame to merge in. Should already be normalized
            (e.g. column names matched to the target).
        on: Primary key columns used for the merge predicate.
        delete_unmatched: When ``True`` (default), rows present in the target
            but not in the source are deleted via
            ``whenNotMatchedBySourceDelete``. Set ``False`` for upsert-only.

    Returns:
        ``None``. The merge is executed in-place on the Delta target.

    Example:
        >>> snapshot_merge("db.table", new_snapshot, on=["id"])
        >>> snapshot_merge("db.table", upserts, on=["id"],
        ...                delete_unmatched=False)
    """
    from delta.tables import DeltaTable

    spark = get_spark()
    pks: List[str] = list(on)
    cols: List[str] = df.columns
    builder: Any = (
        DeltaTable.forName(spark, target_table).alias("t")
        .merge(df.alias("s"), merge_condition(pks))
        .whenMatchedUpdate(set=column_map(cols))
        .whenNotMatchedInsert(values=column_map(cols))
    )
    if delete_unmatched:
        builder = builder.whenNotMatchedBySourceDelete()
    builder.execute()


def cdf_merge(
    target_table: str,
    cdf: "DataFrame",
    on: Iterable[str],
    *,
    metadata_cols: Set[str] = DEFAULT_CDF_METADATA,
) -> bool:
    """Apply a Delta Change Data Feed DataFrame onto ``target_table``.

    Behavior:

    1. Filter ``change_type == 'update_preimage'`` (Delta emits both pre- and
       post- images for updates; the post-image is the desired state).
    2. Short-circuit and return ``False`` if no rows remain (empty window).
    3. Dedupe by primary key, keeping the latest ``commit_version`` /
       ``commit_timestamp`` per key — coalesces multiple changes per key
       within the window.
    4. Merge into the target: ``whenMatchedDelete`` for deletes,
       ``whenMatchedUpdate`` for updates, ``whenNotMatchedInsert`` for
       inserts. Metadata columns (``change_type`` etc.) are stripped from
       data writes.

    Args:
        target_table: Fully-qualified managed Delta table name.
        cdf: DataFrame from ``spark.read.format("delta").option(
            "readChangeFeed", "true").option("startingVersion", N).load(...)``.
            Use :func:`read_cdf` for a thin wrapper.
        on: Primary key columns for merge predicate and dedup window.
        metadata_cols: Set of column names to exclude from the data write.
            Defaults to :data:`DEFAULT_CDF_METADATA`.

    Returns:
        ``True`` if a merge was applied, ``False`` if the CDF window was
        empty (after filtering ``update_preimage``).

    Raises:
        Whatever ``DeltaTable.merge`` raises — typically when CDF is not
        enabled on the source table, or columns are missing.

    Example:
        >>> cdf = read_cdf("abfss://container@acct.../path/to/table/", start_version=last + 1)
        >>> applied = cdf_merge("db.table", cdf, on=["id"])
    """
    from delta.tables import DeltaTable
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    spark = get_spark()
    pks: List[str] = list(on)
    filtered = cdf.where(F.col("change_type") != "update_preimage")
    if not filtered.take(1):
        return False

    window = Window.partitionBy(*pks).orderBy(
        F.desc("commit_version"), F.desc("commit_timestamp"),
    )
    latest = (
        filtered.withColumn("__rn", F.row_number().over(window))
                .where(F.col("__rn") == 1)
                .drop("__rn")
    )
    data_cols: List[str] = [
        c for c in latest.columns if c not in metadata_cols
    ]
    (
        DeltaTable.forName(spark, target_table).alias("t")
        .merge(latest.alias("s"), merge_condition(pks))
        .whenMatchedDelete(condition="s.change_type = 'delete'")
        .whenMatchedUpdate(
            condition="s.change_type <> 'delete'",
            set=column_map(data_cols),
        )
        .whenNotMatchedInsert(
            condition="s.change_type <> 'delete'",
            values=column_map(data_cols),
        )
        .execute()
    )
    return True


def read_cdf(src_path: str, start_version: int) -> "DataFrame":
    """Read the Delta change feed at ``src_path`` from ``start_version`` onward.

    A thin wrapper around the standard CDF reader incantation:

    ``spark.read.format("delta").option("readChangeFeed", "true")
    .option("startingVersion", N).load(src_path)``

    Args:
        src_path: Source Delta folder URI.
        start_version: First commit version to include (inclusive).

    Returns:
        A DataFrame of CDF rows including ``change_type``, ``commit_version``,
        ``commit_timestamp`` columns. Raises at action time if CDF is not
        enabled on the source.

    Example:
        >>> cdf = read_cdf("abfss://container@acct.../path/to/table/", start_version=42)
        >>> cdf.show()
    """
    spark = get_spark()
    return (
        spark.read.format("delta")
             .option("readChangeFeed", "true")
             .option("startingVersion", int(start_version))
             .load(src_path)
    )


def _nbutils() -> Any:
    try:
        from notebookutils import mssparkutils
        return mssparkutils
    except ImportError:
        try:
            import mssparkutils  # type: ignore
            return mssparkutils
        except ImportError:
            return None


__all__: List[str] = [
    "DEFAULT_CDF_METADATA",
    "cdf_merge",
    "column_map",
    "current_delta_version",
    "merge_condition",
    "read_cdf",
    "snapshot_merge",
]
