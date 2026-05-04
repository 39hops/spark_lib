"""SparkSession lookup helpers.

The package never creates a SparkSession. Synapse notebooks already provide
one, and local PySpark jobs should register or activate one explicitly.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

_spark: Optional["SparkSession"] = None


def set_spark(session: "SparkSession") -> None:
    """Register the SparkSession used by spark_lib.

    Use this in local scripts/tests or any runtime where Spark does not expose
    an active session. Synapse notebooks usually do not need it because Spark is
    already active before user code runs.
    """
    global _spark
    _spark = session


def get_spark() -> "SparkSession":
    """Return the registered or active SparkSession.

    This intentionally avoids `SparkSession.builder.getOrCreate()` so imports
    do not mutate the runtime or fight Synapse's pre-created session.
    """
    if _spark is not None:
        return _spark

    active: Optional["SparkSession"] = _active_spark_session()
    if active is not None:
        return active

    frame_session: Optional["SparkSession"] = _spark_from_call_stack()
    if frame_session is not None:
        return frame_session

    raise RuntimeError(
        "No active SparkSession found. In Synapse this should usually be "
        "available automatically; otherwise call spark_lib.set_spark(spark) "
        "once before reading, writing, or running Spark jobs."
    )


def _active_spark_session() -> Optional["SparkSession"]:
    """Return Spark's active session without creating one."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    return SparkSession.getActiveSession()


def _spark_from_call_stack() -> Optional["SparkSession"]:
    """Find a notebook/global `spark` symbol as a Synapse-friendly fallback."""
    for frame_info in inspect.stack()[2:]:
        frame = frame_info.frame
        candidate: Any = frame.f_locals.get("spark", frame.f_globals.get("spark"))
        if _looks_like_spark(candidate):
            return candidate
    return None


def _looks_like_spark(value: Any) -> bool:
    """Duck-type SparkSession to avoid importing PySpark during package import."""
    return (
        value is not None
        and hasattr(value, "read")
        and hasattr(value, "table")
        and hasattr(value, "sparkContext")
    )


__all__ = ["get_spark", "set_spark"]
