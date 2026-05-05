"""Data-dictionary scan over registered Spark databases.

Walks one or more databases registered in the active Spark catalog
(Hive metastore on Synapse), reads each table's column metadata via
``spark.catalog.listColumns`` and table-level properties (location, format)
via ``DESCRIBE TABLE EXTENDED``, and returns one row per column in a
long-format DataFrame suitable for filtering, joining, and reporting.

Public API
----------
- ``scan_databases(dbs, *, max_workers=8, pool=None)`` — return the
  dictionary DataFrame for a set of databases.
- ``write_dictionary(dbs, target_table="db.__data_dictionary", ...)`` —
  scan and overwrite a managed Delta table with the result.

Output schema
-------------
``db, table, table_location, table_format, column_name, ordinal,
data_type, nullable, is_partition, comment``

One row per (db, table, column). ``ordinal`` is 0-based following
``spark.catalog.listColumns`` order.

Filters applied by default
--------------------------
- Table names beginning with ``__`` (state/internal tables).
- Views (``tableType == "VIEW"``).
- Temporary tables.

Conventions
-----------
- Per-table metadata reads run in parallel via :func:`spark_lib.cleanup.run_parallel`
  so a scan over hundreds of tables is not bottlenecked on serial metastore
  round-trips. Failures are logged one-line and skipped (the dictionary is
  still produced for the tables that succeeded).
- ``DESCRIBE TABLE EXTENDED`` is parsed for ``Location`` and ``Provider``
  only; we do not pull row counts or stats here. Profiling is a separate
  concern and belongs in a follow-on module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .cleanup import log, run_parallel
from .session import get_spark

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


_DICTIONARY_SCHEMA: List[Tuple[str, str]] = [
    ("db", "string"),
    ("table", "string"),
    ("table_location", "string"),
    ("table_format", "string"),
    ("column_name", "string"),
    ("ordinal", "int"),
    ("data_type", "string"),
    ("nullable", "boolean"),
    ("is_partition", "boolean"),
    ("comment", "string"),
]


def scan_databases(
    dbs: Iterable[str],
    *,
    max_workers: int = 8,
    pool: Optional[str] = None,
) -> "DataFrame":
    """Scan ``dbs`` and return a long-format data dictionary DataFrame.

    For each database in ``dbs``, lists the registered tables, filters out
    views, temporary tables, and ``__``-prefixed internal tables, then for
    every remaining table reads:

    - column metadata via ``spark.catalog.listColumns`` (name, dataType,
      nullable, isPartition, description),
    - table location and provider via ``DESCRIBE TABLE EXTENDED``.

    Per-table reads run concurrently via :func:`run_parallel`. Tables that
    fail to read are logged and skipped — the returned DataFrame still
    contains every table that succeeded.

    Args:
        dbs: Database names to scan. Iterated once; safe to pass a generator.
        max_workers: Thread count for the per-table metadata reads.
            Defaults to 8.
        pool: Optional Spark FAIR scheduler pool name passed to
            :func:`run_parallel`.

    Returns:
        DataFrame with one row per (db, table, column) and columns
        ``db, table, table_location, table_format, column_name, ordinal,
        data_type, nullable, is_partition, comment``. Empty DataFrame
        (with the same schema) if no tables qualify.

    Example:
        >>> df = scan_databases(["db"])
        >>> df.where("data_type LIKE 'decimal%'").show()
    """
    spark: Any = get_spark()
    db_list: List[str] = list(dbs)
    jobs: List[Dict[str, Any]] = []
    for db in db_list:
        for entry in spark.catalog.listTables(db):
            name: str = str(entry.name)
            if name.startswith("__"):
                continue
            if bool(getattr(entry, "isTemporary", False)):
                continue
            if str(getattr(entry, "tableType", "")).upper() == "VIEW":
                continue
            jobs.append({"name": f"{db}.{name}", "db": db, "table": name})

    log.info(
        "scan_databases: %d tables across %d dbs", len(jobs), len(db_list),
    )
    results: List[Any] = run_parallel(
        _scan_table, jobs, max_workers=max_workers, pool=pool, fail_fast=False,
    )

    rows: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, BaseException):
            continue
        rows.extend(r)

    schema: str = ", ".join(f"{n} {t}" for n, t in _DICTIONARY_SCHEMA)
    if not rows:
        return spark.createDataFrame([], schema)
    return spark.createDataFrame(rows, schema)


def write_dictionary(
    dbs: Iterable[str],
    target_table: str = "db.__data_dictionary",
    *,
    max_workers: int = 8,
    pool: Optional[str] = None,
) -> "DataFrame":
    """Scan ``dbs`` and overwrite ``target_table`` with the result.

    Convenience wrapper around :func:`scan_databases` that persists the
    output to a managed Delta table for downstream querying. Uses
    ``overwriteSchema=true`` so column changes (e.g. when this module is
    extended) propagate without manual migration.

    Args:
        dbs: Database names to scan.
        target_table: Fully-qualified managed Delta table to overwrite.
            Defaults to ``"db.__data_dictionary"`` (note the ``__`` prefix
            so the table is excluded from future self-scans).
        max_workers: Thread count for the per-table metadata reads.
        pool: Optional Spark FAIR scheduler pool name.

    Returns:
        The DataFrame that was written.

    Example:
        >>> write_dictionary(["db"])
        >>> spark.table("db.__data_dictionary").show()
    """
    df = scan_databases(dbs, max_workers=max_workers, pool=pool)
    (
        df.write.format("delta")
          .mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(target_table)
    )
    log.info("write_dictionary: wrote %s", target_table)
    return df


def _scan_table(db: str, table: str, **_: Any) -> List[Dict[str, Any]]:
    """Read column + table metadata for a single (db, table) pair.

    Accepts ``**_`` to absorb the ``name`` key that :func:`run_parallel`
    threads through for logging.
    """
    spark: Any = get_spark()
    qualified: str = f"`{db}`.`{table}`"
    location, fmt = _table_props(qualified)
    cols: List[Any] = list(spark.catalog.listColumns(table, db))
    return [
        {
            "db": db,
            "table": table,
            "table_location": location,
            "table_format": fmt,
            "column_name": str(c.name),
            "ordinal": i,
            "data_type": str(c.dataType),
            "nullable": bool(c.nullable),
            "is_partition": bool(c.isPartition),
            "comment": (str(c.description) if c.description is not None else None),
        }
        for i, c in enumerate(cols)
    ]


def _table_props(qualified: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``Location`` and ``Provider`` out of ``DESCRIBE TABLE EXTENDED``.

    The ``DESCRIBE TABLE EXTENDED`` result has three sections separated by
    blank rows: column list, optional ``# Partition Information``, and
    ``# Detailed Table Information``. We walk to the third section and
    pull the two property rows we care about.
    """
    spark: Any = get_spark()
    location: Optional[str] = None
    fmt: Optional[str] = None
    in_details: bool = False
    for row in spark.sql(f"DESCRIBE TABLE EXTENDED {qualified}").collect():
        key: str = (row["col_name"] or "").strip()
        value: str = (row["data_type"] or "").strip() if "data_type" in row else ""
        if not in_details:
            if key.startswith("# Detailed Table Information"):
                in_details = True
            continue
        if key == "Location":
            location = value or None
        elif key == "Provider":
            fmt = value or None
    return location, fmt


__all__: List[str] = [
    "scan_databases",
    "write_dictionary",
]
