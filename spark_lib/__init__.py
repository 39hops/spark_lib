"""Importable package for Synapse notebook transform helpers."""
from __future__ import annotations

from .cleanup import clean_columns, dedupe, quiet_azure_logging, run_parallel
from .delta import (
    DEFAULT_CDF_METADATA,
    cdf_merge,
    column_map,
    current_delta_version,
    merge_condition,
    read_cdf,
    snapshot_merge,
)
from .matching import (
    fill_missing_from_match,
    fuzzy_match,
    infer_key_from_text,
    ml_fuzzy_match,
    normalize_text,
    search_database,
)
from .session import get_spark, set_spark
from .sync import SyncResult, SyncSpec, SyncState, run_sync, sync_delta_to_table
from .transforms import Input, Output, transform, transform_df

__all__ = [
    "DEFAULT_CDF_METADATA",
    "Input",
    "Output",
    "SyncResult",
    "SyncSpec",
    "SyncState",
    "cdf_merge",
    "clean_columns",
    "column_map",
    "current_delta_version",
    "dedupe",
    "fill_missing_from_match",
    "fuzzy_match",
    "get_spark",
    "infer_key_from_text",
    "merge_condition",
    "ml_fuzzy_match",
    "normalize_text",
    "quiet_azure_logging",
    "read_cdf",
    "run_parallel",
    "run_sync",
    "search_database",
    "set_spark",
    "snapshot_merge",
    "sync_delta_to_table",
    "transform",
    "transform_df",
]
