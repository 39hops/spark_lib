"""Delta Lake primitives: version lookup, snapshot merge, CDF merge.

These helpers wrap the small set of `DeltaTable` boilerplate used across
ingestion notebooks. They intentionally do nothing project-specific (no path
conventions, no state tables) so they compose cleanly.
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
    """Build a SQL merge predicate joining target and source on `pks`."""
    return " AND ".join(
        f"{target_alias}.{c} = {source_alias}.{c}" for c in pks
    )


def column_map(cols: Iterable[str], prefix: str = "s") -> Dict[str, str]:
    """`{col: f"{prefix}.{col}"}` for use with `whenMatchedUpdate(set=...)`."""
    return {c: f"{prefix}.{c}" for c in cols}


def current_delta_version(path: str) -> Optional[int]:
    """Cheap file-only lookup of the latest Delta commit at `path`.

    Reads `_delta_log/_last_checkpoint` first and falls back to listing the
    log directory. Returns `None` when neither is readable so callers can
    decide whether to keep going. Avoids `DeltaTable.forPath` URI strictness
    and `DESCRIBE HISTORY` scans.

    Currently uses `notebookutils`/`mssparkutils` for filesystem access.
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
    """Upsert `df` into `target_table`, optionally deleting rows in target
    that aren't present in source.

    Use `delete_unmatched=True` to make the target a true mirror of the source
    snapshot (preserving Delta time travel). The target must already exist.
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
    """Apply a Delta CDF DataFrame onto `target_table`.

    `cdf` should be the result of `spark.read.format("delta").option(
    "readChangeFeed", "true")...`. The function dedupes by primary key
    (latest `commit_version`/`commit_timestamp` wins), filters
    `update_preimage`, and applies inserts/updates/deletes to the target.
    Returns `True` when at least one change was applied, `False` when CDF
    yielded an empty window.
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
    """Read the Delta change feed at `src_path` from `start_version` onward."""
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
