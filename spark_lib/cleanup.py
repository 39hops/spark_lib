"""Column cleanup, deduplication, logging, and parallel Spark job helpers."""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from queue import Empty, Queue
from threading import Lock, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .session import get_spark

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


# ---------- logger ---------------------------------------------------------

log: logging.Logger = logging.getLogger("synapse.cleanup")
if not log.handlers:
    _handler: logging.Handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


_NOISY_LOGGERS: List[str] = [
    "py4j",
    "py4j.java_gateway",
    "py4j.clientserver",
    "azure",
    "azure.core",
    "azure.identity",
    "azure.core.pipeline.policies.http_logging_policy",
    "msal",
    "msrest",
    "urllib3",
    "adal-python",
]


def quiet_azure_logging(level: int = logging.WARNING) -> None:
    """Raise noisy Azure and py4j loggers above normal notebook chatter.

    Examples:
        Default — silence to WARNING:

        >>> quiet_azure_logging()

        Silence harder — only show errors:

        >>> import logging
        >>> quiet_azure_logging(level=logging.ERROR)
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    log.info(
        "silenced %d noisy loggers (level=%s)",
        len(_NOISY_LOGGERS),
        logging.getLevelName(level),
    )


# ---------- column normalization ------------------------------------------

_NON_ALNUM_RE: "re.Pattern[str]" = re.compile(r"[^a-z0-9]+")
_LEAD_DIGIT_RE: "re.Pattern[str]" = re.compile(r"^\d")


def _to_snake(name: str) -> str:
    """Lower, strip accents, keep [a-z0-9], and collapse runs to `_`."""
    nfkd: str = unicodedata.normalize("NFKD", name)
    ascii_only: str = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned: str = _NON_ALNUM_RE.sub("_", ascii_only.lower()).strip("_")
    if not cleaned:
        return "_"
    if _LEAD_DIGIT_RE.match(cleaned):
        cleaned = "_" + cleaned
    return cleaned


def clean_columns(df: "DataFrame") -> "DataFrame":
    """Rename every column to snake_case ASCII with collision suffixes.

    Strips accents, lowercases, replaces non-alphanumeric runs with ``_``,
    prepends ``_`` to names that would start with a digit, and disambiguates
    duplicate normalized names by appending ``_1``, ``_2``, ...

    Examples:
        >>> df.columns
        ['ID', 'Value ($)', 'Façade']
        >>> clean_columns(df).columns
        ['id', 'value', 'facade']

        Duplicates get suffixed:

        >>> df2.columns
        ['name', 'Name']
        >>> clean_columns(df2).columns
        ['name', 'name_1']

        Leading digits are escaped:

        >>> df3.columns
        ['1st_quarter']
        >>> clean_columns(df3).columns
        ['_1st_quarter']
    """
    seen: Dict[str, int] = {}
    new_names: List[str] = []
    changed: int = 0
    for original in df.columns:
        base: str = _to_snake(original)
        count: int = seen.get(base, 0)
        unique: str = f"{base}_{count}" if count else base
        seen[base] = count + 1
        new_names.append(unique)
        if unique != original:
            changed += 1
    if changed == 0:
        return df
    log.info("clean_columns: renamed %d/%d columns", changed, len(df.columns))
    return df.toDF(*new_names)


# ---------- deduplication --------------------------------------------------

def dedupe(
    df: "DataFrame",
    pks: Union[str, Iterable[str]],
    order_by: Union[str, Iterable[str]],
    descending: bool = True,
) -> "DataFrame":
    """Keep one row per primary-key group, picked by ``order_by``.

    Implemented with ``row_number()`` over a window partitioned by ``pks``
    and ordered by ``order_by``. By default ``descending=True`` so the
    "latest" row wins for timestamp-style ordering keys.

    Examples:
        Latest row per ``id`` by ``updated_at``:

        >>> dedupe(df, pks="id", order_by="updated_at")

        Composite PK, multi-column ordering:

        >>> dedupe(df, pks=["id", "fk_id"],
        ...       order_by=["updated_at", "version"])

        Earliest row instead of latest:

        >>> dedupe(df, pks="id", order_by="created_at", descending=False)
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    pk_list: List[str] = [pks] if isinstance(pks, str) else list(pks)
    order_cols: List[str] = (
        [order_by] if isinstance(order_by, str) else list(order_by)
    )
    order_exprs: List["Column"] = [
        F.desc(c) if descending else F.asc(c) for c in order_cols
    ]
    window = Window.partitionBy(*pk_list).orderBy(*order_exprs)
    rn: str = "__rn_dedupe"
    log.info(
        "dedupe: pks=%s order_by=%s desc=%s", pk_list, order_cols, descending
    )
    return (
        df.withColumn(rn, F.row_number().over(window))
          .where(F.col(rn) == 1)
          .drop(rn)
    )


# ---------- parallel job runner --------------------------------------------

T = TypeVar("T")


def run_parallel(
    fn: Callable[..., T],
    jobs: List[Dict[str, Any]],
    max_workers: int = 4,
    pool: Optional[str] = None,
    fail_fast: bool = False,
) -> List[Union[T, BaseException]]:
    """Run ``fn(**job)`` for every job concurrently in worker threads.

    Each job is a dict whose keys map to ``fn``'s keyword arguments. An
    optional ``"name"`` key is used in log lines. Failures are returned as
    exception objects in the result list (not raised), unless
    ``fail_fast=True`` in which case the first failure cancels the rest.

    This parallelizes submissions from the driver. Spark only spreads work
    across executors when ``fn`` launches distributed Spark actions; driver-
    bound catalog/DDL calls can still appear to use one executor or no
    executor work at all.

    Examples:
        Run a function across many inputs:

        >>> def load(name, db, **_):
        ...     return spark.read.table(f"{db}.{name}").count()
        >>> jobs = [{"name": t, "db": "db"} for t in ["table_a", "table_b"]]
        >>> results = run_parallel(load, jobs, max_workers=4)

        Use a Spark FAIR pool so the parallel work has its own scheduling lane:

        >>> run_parallel(load, jobs, max_workers=8, pool="ingest")

        Fail-fast variant (first exception cancels remaining work):

        >>> run_parallel(load, jobs, fail_fast=True)
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    spark: Any = get_spark()
    sc: Any = spark.sparkContext
    if pool is not None:
        _warn_if_pool_without_fair(sc, pool)
    results: List[Any] = [None] * len(jobs)

    def _worker(idx: int, job: Dict[str, Any]) -> Tuple[int, Any]:
        previous_pool: Optional[str] = None
        if pool is not None:
            previous_pool = sc.getLocalProperty("spark.scheduler.pool")
            sc.setLocalProperty("spark.scheduler.pool", pool)
        name: str = str(job.get("name", f"job_{idx}"))
        t0: float = time.monotonic()
        log.info("[%s] start", name)
        try:
            value: Any = fn(**job)
        except BaseException as exc:
            elapsed: float = time.monotonic() - t0
            if fail_fast:
                log.exception("[%s] failed after %.1fs", name, elapsed)
                raise
            log.warning(
                "[%s] failed after %.1fs: %s: %s",
                name, elapsed, type(exc).__name__, exc,
            )
            log.debug("[%s] traceback", name, exc_info=exc)
            return idx, exc
        finally:
            if pool is not None:
                sc.setLocalProperty("spark.scheduler.pool", previous_pool)
        log.info("[%s] done in %.1fs", name, time.monotonic() - t0)
        return idx, value

    log.info(
        "run_parallel: %d jobs, max_workers=%d, pool=%s",
        len(jobs),
        max_workers,
        pool,
    )
    _run_worker_threads(
        _worker,
        jobs,
        results,
        max_workers=max_workers,
        fail_fast=fail_fast,
        spark=spark,
    )

    failures: int = sum(1 for r in results if isinstance(r, BaseException))
    if failures:
        log.warning("run_parallel: %d/%d jobs failed", failures, len(jobs))
    return results


def drop_database_tables(
    database: str,
    *,
    tables: Optional[Iterable[str]] = None,
    include_views: bool = False,
    max_workers: int = 8,
    pool: Optional[str] = None,
    dry_run: bool = False,
) -> List[Union[str, BaseException]]:
    """Drop managed/external tables in ``database`` concurrently.

    The concurrency here is driver-side SQL submission. ``DROP TABLE`` is
    catalog/DDL-heavy, so it may not consume all executors even when several
    drop statements are submitted in parallel.

    Args:
        database: Database/schema to clean.
        tables: Optional table-name allowlist. Names are unqualified.
        include_views: When ``True``, also drop views from the database.
        max_workers: Thread count passed to :func:`run_parallel`.
        pool: Optional Spark FAIR scheduler pool.
        dry_run: Return the objects that would be dropped without executing.

    Returns:
        Ordered ``List[str | BaseException]`` matching :func:`run_parallel`.
        Successful entries are fully-qualified object names.

    Examples:
        Drop everything in a database:

        >>> drop_database_tables("db")

        Drop only specific tables, including views:

        >>> drop_database_tables(
        ...     "db",
        ...     tables=["table_a", "table_b"],
        ...     include_views=True,
        ... )

        Preview without executing:

        >>> drop_database_tables("db", dry_run=True)
    """
    spark: Any = get_spark()
    selected: Optional[Set[str]] = set(tables) if tables is not None else None
    jobs: List[Dict[str, Any]] = []
    for entry in spark.catalog.listTables(database):
        name: str = str(entry.name)
        if bool(getattr(entry, "isTemporary", False)):
            continue
        if selected is not None and name not in selected:
            continue

        table_type: str = str(getattr(entry, "tableType", "")).upper()
        is_view: bool = table_type == "VIEW"
        if is_view and not include_views:
            continue

        kind: str = "VIEW" if is_view else "TABLE"
        qualified: str = f"{database}.{name}"
        jobs.append({
            "name": qualified,
            "object_name": qualified,
            "statement": (
                f"DROP {kind} IF EXISTS "
                f"{_qualified_identifier(database, name)}"
            ),
        })

    def _drop_table(object_name: str, statement: str, name: str = "") -> str:
        if dry_run:
            log.info("[dry-run] %s", statement)
            return object_name
        spark.sql(statement)
        return object_name

    return run_parallel(
        _drop_table,
        jobs,
        max_workers=max_workers,
        pool=pool,
        fail_fast=False,
    )


def _qualified_identifier(database: str, table: str) -> str:
    return f"{_quote_identifier(database)}.{_quote_identifier(table)}"


def _run_worker_threads(
    worker: Callable[[int, Dict[str, Any]], Tuple[int, Any]],
    jobs: List[Dict[str, Any]],
    results: List[Any],
    *,
    max_workers: int,
    fail_fast: bool,
    spark: Any,
) -> None:
    job_queue: "Queue[Tuple[int, Dict[str, Any]]]" = Queue()
    for item in enumerate(jobs):
        job_queue.put(item)

    first_failure: List[BaseException] = []
    failure_lock: Lock = Lock()

    def _thread_main() -> None:
        while True:
            if fail_fast and first_failure:
                return
            try:
                idx, job = job_queue.get_nowait()
            except Empty:
                return
            try:
                result_idx, value = worker(idx, job)
                results[result_idx] = value
            except BaseException as exc:
                if fail_fast:
                    with failure_lock:
                        if not first_failure:
                            first_failure.append(exc)
                    return
                results[idx] = exc
            finally:
                job_queue.task_done()

    thread_cls: Any = _spark_thread_class()
    threads: List[Any] = []
    for _ in range(min(max_workers, len(jobs))):
        if thread_cls is Thread:
            thread: Any = thread_cls(target=_thread_main)
        else:
            thread = thread_cls(target=_thread_main, session=spark)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if first_failure:
        raise first_failure[0]


def _spark_thread_class() -> Any:
    try:
        from pyspark import InheritableThread
    except Exception:
        return Thread
    return InheritableThread


def _warn_if_pool_without_fair(sc: Any, pool: str) -> None:
    try:
        mode: str = str(sc.getConf().get("spark.scheduler.mode", "FIFO"))
    except Exception:
        return
    if mode.upper() != "FAIR":
        log.warning(
            "pool=%s was requested, but spark.scheduler.mode=%s; "
            "Spark will not schedule jobs across FAIR pools unless FAIR mode "
            "is configured when the session starts",
            pool,
            mode,
        )


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


__all__: List[str] = [
    "clean_columns",
    "dedupe",
    "drop_database_tables",
    "log",
    "quiet_azure_logging",
    "run_parallel",
]
