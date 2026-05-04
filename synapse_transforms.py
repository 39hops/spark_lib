"""
synapse_transforms — a tiny `transforms.api`-style helper for Synapse notebooks.

Drop this whole file into a notebook cell (or %run it from a workspace file)
and you get:

    from synapse_transforms import Input, Output, transform, transform_df

    @transform_df(
        output  = Output.table("sales.daily_orders", mode="overwrite"),
        orders  = Input("abfss://raw@acct.dfs.core.windows.net/orders/"),  # delta dir
        cust    = Input("abfss://raw@acct.dfs.core.windows.net/cust.csv"), # csv
        regions = Input.table("ref.regions"),                              # managed table
    )
    def daily_orders(orders, cust, regions):
        return (orders.join(cust, "customer_id")
                      .join(regions, "region_id"))

    daily_orders()            # runs the job

Format is inferred from the path extension (delta / parquet / csv / tsv /
json / orc / avro / xlsx). Override with `format=`. Anything that isn't an
abfss:// URI is treated as a managed-table reference.

Assumes the Synapse-provided `spark` SparkSession already exists in the
notebook namespace — we reference it directly. xlsx I/O additionally needs
pandas + openpyxl. xlsx round-trips through `notebookutils.fs` to /tmp.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    # Synapse pre-creates this in the notebook namespace; declared here only
    # so type checkers know the symbol. At runtime it resolves to the
    # notebook's `spark` global.
    spark: "SparkSession"


# ---------- helpers ---------------------------------------------------------

PathLike = str
PartitionLike = Union[str, Iterable[str]]


def _is_abfss(path: PathLike) -> bool:
    return path.startswith("abfss://")


_FORMAT_BY_EXT: Dict[str, str] = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".xlsx": "excel",
    ".xls": "excel",
    ".orc": "orc",
    ".avro": "avro",
    ".delta": "delta",
}

# Compression suffixes Spark appends to part-files (e.g. .snappy.parquet).
# We strip these before matching the data extension.
_COMPRESSION_SUFFIXES: Tuple[str, ...] = (
    ".gz", ".snappy", ".zstd", ".zst", ".lz4", ".bz2", ".deflate",
)


def _infer_format(path: PathLike) -> Optional[str]:
    """Guess a Spark format from a path string alone (no I/O).

    Returns None for an abfss directory with no recognizable extension —
    callers that can afford a metadata round-trip should fall back to
    `_peek_format` to inspect the directory contents.
    """
    ext: str = os.path.splitext(path.lower().rstrip("/"))[1]
    if ext in _FORMAT_BY_EXT:
        return _FORMAT_BY_EXT[ext]
    if not _is_abfss(path):           # "db.table" or "cat.db.table"
        return "table"
    return None


def _peek_format(path: PathLike) -> Optional[str]:
    """Inspect an abfss directory's contents to infer its format.

    Looks for a `_delta_log/` child (delta), otherwise the extension of
    the first `part-*` data file. Returns None if neither is found or
    the listing fails.
    """
    try:
        entries: List[Any] = list(_nbutils().fs.ls(path))
    except Exception:                 # noqa: BLE001 — peek is best-effort
        return None

    names: List[str] = [
        getattr(e, "name", str(e)).rstrip("/") for e in entries
    ]
    if any(n == "_delta_log" for n in names):
        return "delta"

    for n in names:
        if not n.startswith("part-"):
            continue
        lower: str = n.lower()
        for comp in _COMPRESSION_SUFFIXES:
            if lower.endswith(comp):
                lower = lower[: -len(comp)]
                break
        ext: str = os.path.splitext(lower)[1]
        if ext in _FORMAT_BY_EXT:
            return _FORMAT_BY_EXT[ext]
    return None


def _as_list(x: PartitionLike) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


def _local_tmp(path: PathLike) -> str:
    base: str = os.path.basename(path.rstrip("/")) or "_tmp"
    return "/tmp/" + base


def _nbutils() -> Any:
    """Synapse's filesystem helper. Module name varies across runtimes."""
    try:
        from notebookutils import mssparkutils  # newer Synapse runtimes
        return mssparkutils
    except ImportError:
        try:
            import mssparkutils  # type: ignore  # older runtimes
            return mssparkutils
        except ImportError:
            raise RuntimeError(
                "notebookutils / mssparkutils not available — "
                "needed for xlsx round-trips through ADLS."
            )


# ---------- Input -----------------------------------------------------------

@dataclass
class Input:
    """A lazy reference to a dataset. Call `.dataframe()` to materialize."""

    path: PathLike
    format: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def table(cls, name: str, **options: Any) -> "Input":
        return cls(path=name, format="table", options=dict(options))

    @property
    def fmt(self) -> str:
        if self.format:
            return self.format
        inferred: Optional[str] = _infer_format(self.path)
        if inferred is not None:
            return inferred
        # bare abfss directory — peek inside to tell delta from parquet/csv/...
        peeked: Optional[str] = _peek_format(self.path)
        return peeked or "delta"

    def dataframe(self) -> "DataFrame":
        fmt: str = self.fmt

        if fmt == "table":
            return spark.table(self.path)
        if fmt == "excel":
            return _read_excel(self.path, **self.options)

        opts: Dict[str, Any] = dict(self.options)
        reader: Any = spark.read.format(fmt)

        if fmt == "csv":
            reader = reader.option("header", opts.pop("header", "true"))
            reader = reader.option("inferSchema", opts.pop("inferSchema", "true"))
            if self.path.lower().endswith(".tsv"):
                reader = reader.option("sep", opts.pop("sep", "\t"))

        for k, v in opts.items():
            reader = reader.option(k, v)
        return reader.load(self.path)

    # transforms.api compat alias
    def read(self) -> "DataFrame":
        return self.dataframe()


# ---------- Output ----------------------------------------------------------

@dataclass
class Output:
    """A sink. Call `.write(df)` (or use a `@transform_df` decorator)."""

    path: PathLike
    format: Optional[str] = None
    mode: str = "overwrite"
    partition_by: Optional[PartitionLike] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def table(
        cls,
        name: str,
        *,
        mode: str = "overwrite",
        format: str = "delta",
        partition_by: Optional[PartitionLike] = None,
        **options: Any,
    ) -> "Output":
        return cls(
            path=name,
            format="table",
            mode=mode,
            partition_by=partition_by,
            options={"_table_format": format, **options},
        )

    @property
    def fmt(self) -> str:
        # Outputs don't peek — destination may not exist yet. Default to delta
        # for an unmarked abfss directory, which matches Synapse convention.
        return self.format or _infer_format(self.path) or "delta"

    def write(self, df: "DataFrame") -> None:
        fmt: str = self.fmt

        if fmt == "excel":
            _write_excel(df, self.path, **self.options)
            return

        opts: Dict[str, Any] = dict(self.options)

        if fmt == "table":
            tbl_fmt: str = opts.pop("_table_format", "delta")
            writer: Any = df.write.format(tbl_fmt).mode(self.mode)
            for k, v in opts.items():
                writer = writer.option(k, v)
            if self.partition_by is not None:
                writer = writer.partitionBy(*_as_list(self.partition_by))
            writer.saveAsTable(self.path)
            return

        writer = df.write.format(fmt).mode(self.mode)
        if fmt == "csv":
            writer = writer.option("header", opts.pop("header", "true"))
        for k, v in opts.items():
            writer = writer.option(k, v)
        if self.partition_by is not None:
            writer = writer.partitionBy(*_as_list(self.partition_by))
        writer.save(self.path)

    # transforms.api compat alias
    def write_dataframe(self, df: "DataFrame") -> None:
        self.write(df)

    def merge_into(
        self,
        df: "DataFrame",
        on: PartitionLike,
        when_matched_update: bool = True,
        when_not_matched_insert: bool = True,
    ) -> None:
        """Delta MERGE — upsert by key(s). Target must already exist."""
        from delta.tables import DeltaTable

        keys: List[str] = _as_list(on)
        target: Any
        if self.fmt == "table":
            target = DeltaTable.forName(spark, self.path)
        else:
            target = DeltaTable.forPath(spark, self.path)

        cond: str = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        builder: Any = target.alias("t").merge(df.alias("s"), cond)
        if when_matched_update:
            builder = builder.whenMatchedUpdateAll()
        if when_not_matched_insert:
            builder = builder.whenNotMatchedInsertAll()
        builder.execute()


# ---------- excel via pandas ------------------------------------------------

def _read_excel(
    path: PathLike,
    sheet_name: Union[str, int] = 0,
    **kwargs: Any,
) -> "DataFrame":
    import pandas as pd

    if _is_abfss(path):
        local: str = _local_tmp(path)
        _nbutils().fs.cp(path, "file://" + local, recurse=False)
        pdf = pd.read_excel(local, sheet_name=sheet_name, **kwargs)
    else:
        pdf = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    return spark.createDataFrame(pdf)


def _write_excel(
    df: "DataFrame",
    path: PathLike,
    sheet_name: str = "Sheet1",
    **kwargs: Any,
) -> None:
    pdf = df.toPandas()
    if _is_abfss(path):
        local: str = _local_tmp(path)
        pdf.to_excel(local, sheet_name=sheet_name, index=False, **kwargs)
        _nbutils().fs.cp("file://" + local, path, recurse=False)
    else:
        pdf.to_excel(path, sheet_name=sheet_name, index=False, **kwargs)


# ---------- decorators ------------------------------------------------------

TransformDFFn = Callable[..., "DataFrame"]
TransformFn = Callable[..., Any]
RunDFFn = Callable[[], "DataFrame"]
RunFn = Callable[[], Any]


def transform_df(
    *,
    output: Output,
    **inputs: Input,
) -> Callable[[TransformDFFn], RunDFFn]:
    """Compute fn returns a DataFrame; we write it to `output`.

    Function param names must match the input keyword names.
    """
    _validate(output, inputs)

    def deco(fn: TransformDFFn) -> RunDFFn:
        @wraps(fn)
        def run() -> "DataFrame":
            kwargs: Dict[str, "DataFrame"] = {
                name: inp.dataframe() for name, inp in inputs.items()
            }
            result: Optional["DataFrame"] = fn(**kwargs)
            if result is None:
                raise ValueError(
                    f"{fn.__name__} returned None — "
                    "transform_df expects a DataFrame."
                )
            output.write(result)
            return result

        # transforms.api compat: expose .compute(), .inputs, .output
        setattr(run, "compute", run)
        setattr(run, "inputs", inputs)
        setattr(run, "output", output)
        return run

    return deco


def transform(
    *,
    output: Optional[Output] = None,
    outputs: Optional[Mapping[str, Output]] = None,
    **inputs: Input,
) -> Callable[[TransformFn], RunFn]:
    """Lower-level: hand the raw Input/Output objects to the function.

    Use when you need multiple outputs, custom write modes (e.g. delta MERGE),
    or want to read the same Input multiple times.
    """
    if output is not None and outputs is not None:
        raise ValueError("Pass either `output=` or `outputs=`, not both.")

    sinks: Dict[str, Output] = (
        {"output": output} if output is not None else dict(outputs or {})
    )
    _validate_dict(sinks, inputs)

    def deco(fn: TransformFn) -> RunFn:
        @wraps(fn)
        def run() -> Any:
            return fn(**inputs, **sinks)

        setattr(run, "compute", run)
        setattr(run, "inputs", inputs)
        setattr(run, "outputs", sinks)
        return run

    return deco


def _validate(output: Output, inputs: Mapping[str, Input]) -> None:
    if not isinstance(output, Output):
        raise TypeError("output= must be an Output")
    _validate_dict({"output": output}, inputs)


def _validate_dict(
    outputs: Mapping[str, Output],
    inputs: Mapping[str, Input],
) -> None:
    for n, o in outputs.items():
        if not isinstance(o, Output):
            raise TypeError(f"output {n!r} is not an Output")
    for n, i in inputs.items():
        if not isinstance(i, Input):
            raise TypeError(f"input {n!r} is not an Input")


__all__: List[str] = ["Input", "Output", "transform", "transform_df"]
