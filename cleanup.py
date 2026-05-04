"""
cleanup — column normalization, deduplication, parallel job runner, and
log silencing for Synapse Spark notebooks.

Paste/`%run` alongside `synapse_transforms.py`. Assumes the Synapse-provided
`spark` SparkSession exists in the notebook namespace.

    from cleanup import clean_columns, dedupe, run_parallel, quiet_azure_logging

    quiet_azure_logging()                                  # once, at top of nb

    df = clean_columns(raw_df)                             # snake_case columns
    df = dedupe(df, pks=["customer_id"], order_by="updated_at")

    def ingest(name: str, src: str, dst: str) -> None:
        ...

    jobs = [
        {"name": "orders",    "src": "abfss://...", "dst": "lab.orders"},
        {"name": "customers", "src": "abfss://...", "dst": "lab.customers"},
    ]
    run_parallel(ingest, jobs, max_workers=8, pool="ingest")

For `run_parallel` the SparkSession must be configured with
`spark.scheduler.mode=FAIR` (set at session start in the Synapse pool config).
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession
    spark: "SparkSession"


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
    """Raise the threshold of noisy Azure / py4j loggers so they stop
    cluttering notebook output. Call once at the top of a notebook."""
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
    """Lower → strip accents → keep [a-z0-9] → collapse runs to single `_`."""
    nfkd: str = unicodedata.normalize("NFKD", name)
    ascii_only: str = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned: str = _NON_ALNUM_RE.sub("_", ascii_only.lower()).strip("_")
    if not cleaned:
        return "_"
    if _LEAD_DIGIT_RE.match(cleaned):
        cleaned = "_" + cleaned
    return cleaned


def clean_columns(df: "DataFrame") -> "DataFrame":
    """Rename every column to snake_case ASCII.

    Collisions after cleaning get a numeric suffix (`_1`, `_2`, …) so we
    never silently drop columns.
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
    """Keep one row per primary-key group, picked by `order_by`.

    `row_number().over(Window.partitionBy(pks).orderBy(order_by))` then
    filters to row_number == 1. `descending=True` (default) keeps the
    *latest* row when `order_by` is a timestamp.
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
    """Run `fn(**job)` for every job concurrently via ThreadPoolExecutor.

    Each worker sets `spark.scheduler.pool=<pool>` so Spark's FAIR scheduler
    can interleave jobs (the SparkSession must already have
    `spark.scheduler.mode=FAIR`).

    Returns one result per job, in input order. If `fail_fast=False`
    (default), exceptions are caught, logged, and returned in their slot;
    if `fail_fast=True`, the first failure is re-raised.
    """
    sc: Any = spark.sparkContext  # noqa: F821 — notebook-provided
    results: List[Any] = [None] * len(jobs)

    def _worker(idx: int, job: Dict[str, Any]) -> Tuple[int, Any]:
        if pool is not None:
            sc.setLocalProperty("spark.scheduler.pool", pool)
        name: str = str(job.get("name", f"job_{idx}"))
        t0: float = time.monotonic()
        log.info("[%s] start", name)
        try:
            value: Any = fn(**job)
        except BaseException as exc:
            log.exception(
                "[%s] failed after %.1fs", name, time.monotonic() - t0
            )
            if fail_fast:
                raise
            return idx, exc
        log.info("[%s] done in %.1fs", name, time.monotonic() - t0)
        return idx, value

    log.info(
        "run_parallel: %d jobs, max_workers=%d, pool=%s",
        len(jobs),
        max_workers,
        pool,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, i, j) for i, j in enumerate(jobs)]
        for fut in as_completed(futures):
            idx, value = fut.result()
            results[idx] = value

    failures: int = sum(1 for r in results if isinstance(r, BaseException))
    if failures:
        log.warning("run_parallel: %d/%d jobs failed", failures, len(jobs))
    return results


__all__: List[str] = [
    "clean_columns",
    "dedupe",
    "log",
    "quiet_azure_logging",
    "run_parallel",
]
