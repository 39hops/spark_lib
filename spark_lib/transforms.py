"""
Tiny `transforms.api`-style helpers for Synapse notebooks.

Use the package import in reusable code:

    from spark_lib import Input, Output, transform, transform_df

Synapse normally exposes an active SparkSession. If your runtime does not,
call `spark_lib.set_spark(spark)` once before executing transforms.
"""
from __future__ import annotations

import hashlib
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

from .session import get_spark

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# ---------- helpers ---------------------------------------------------------

PathLike = str
PartitionLike = Union[str, Iterable[str]]
WriteData = Union["DataFrame", Dict[str, "DataFrame"]]


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

# Spark may write compressed part files such as `part-000.snappy.parquet`.
# When peeking into a folder, strip compression suffixes before checking the
# actual data extension.
_COMPRESSION_SUFFIXES: Tuple[str, ...] = (
    ".gz", ".snappy", ".zstd", ".zst", ".lz4", ".bz2", ".deflate",
)


def _infer_format(path: PathLike) -> Optional[str]:
    """Guess a Spark format from a path string alone.

    This stays pure and cheap. Bare `abfss://` folders return `None` so callers
    can decide whether to spend a metadata call peeking into the directory.
    """
    ext: str = os.path.splitext(path.lower().rstrip("/"))[1]
    if ext in _FORMAT_BY_EXT:
        return _FORMAT_BY_EXT[ext]
    if not _is_abfss(path):
        return "table"
    return None


def _peek_format(path: PathLike) -> Optional[str]:
    """Inspect an abfss directory's contents to infer its format.

    Delta folders are identified by `_delta_log`; otherwise we use the first
    Spark `part-*` file extension. Failures return `None` because peeking is a
    convenience, not a reason to break a notebook before Spark gets a chance to
    read with the default format.
    """
    try:
        entries: List[Any] = list(_nbutils().fs.ls(path))
    except Exception:  # noqa: BLE001 - peek is best-effort
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
    """Map a remote report path to a driver-local temp file path.

    Hashed so concurrent jobs writing to different abfss paths with the same
    basename do not collide on the staging file.
    """
    base: str = os.path.basename(path.rstrip("/")) or "_tmp"
    digest: str = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/{digest}_{base}"


def _nbutils() -> Any:
    """Synapse's filesystem helper. Module name varies across runtimes."""
    try:
        from notebookutils import mssparkutils
        return mssparkutils
    except ImportError:
        try:
            import mssparkutils  # type: ignore
            return mssparkutils
        except ImportError:
            raise RuntimeError(
                "notebookutils / mssparkutils not available; needed for xlsx "
                "round-trips through ADLS."
            )


# ---------- Input -----------------------------------------------------------

@dataclass(init=False)
class Input:
    """A lazy reference to a dataset. Call `.dataframe()` to materialize.

    Format is resolved by:

    1. ``format=`` argument if given.
    2. File extension if the path has one (``.csv``, ``.parquet``, ``.delta``...).
    3. Peeking the abfss directory for ``_delta_log`` or a ``part-*`` file.
    4. Defaulting to ``"delta"``.

    Examples:
        Read a Delta folder (format inferred from ``_delta_log`` peek):

        >>> source = Input("abfss://container@acct.../path/to/table/")
        >>> df = source.dataframe()

        Read a registered managed table:

        >>> table = Input.table("db.table")
        >>> df = table.dataframe()

        Read a CSV with non-default options (forwarded to the reader):

        >>> source = Input(
        ...     "abfss://.../data.csv",
        ...     format="csv",
        ...     header="true",
        ...     inferSchema="true",
        ... )

        Force a format (e.g. JSON folder without ``.json`` extension):

        >>> source = Input("abfss://.../path/to/data/", format="json")
    """

    path: PathLike
    format: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        path: PathLike,
        format: Optional[str] = None,
        **options: Any,
    ) -> None:
        self.path = path
        self.format = format
        self.options = dict(options)
        self._resolved_fmt: Optional[str] = None

    @classmethod
    def table(cls, name: str, **options: Any) -> "Input":
        """Construct an :class:`Input` that reads a registered managed table.

        Examples:
            >>> Input.table("db.table").dataframe()
            >>> Input.table("db.table").read().filter("value > 0")
        """
        return cls(path=name, format="table", **options)

    @property
    def fmt(self) -> str:
        """Resolve the input format, peeking only for unmarked abfss folders.

        Cached so repeated reads (or `inp.fmt` accessed for logging then
        reading) do not trigger multiple `fs.ls` calls.

        Examples:
            >>> Input("data.csv").fmt
            'csv'
            >>> Input("abfss://.../path/to/table/").fmt   # peeks once
            'delta'
            >>> Input("db.table", format="table").fmt
            'table'
        """
        if self._resolved_fmt is not None:
            return self._resolved_fmt
        if self.format:
            self._resolved_fmt = self.format
            return self._resolved_fmt
        inferred: Optional[str] = _infer_format(self.path)
        if inferred is not None:
            self._resolved_fmt = inferred
            return self._resolved_fmt
        peeked: Optional[str] = _peek_format(self.path)
        self._resolved_fmt = peeked or "delta"
        return self._resolved_fmt

    def dataframe(self) -> "DataFrame":
        """Materialize this input as a Spark DataFrame.

        Examples:
            >>> Input.table("db.table").dataframe().count()
            1234
            >>> Input("abfss://.../path/to/table/").dataframe().columns
            ['id', 'value', 'updated_at']
        """
        spark = get_spark()
        fmt: str = self.fmt

        if fmt == "table":
            return spark.table(self.path)
        if fmt == "excel":
            return _read_excel(self.path, **self.options)

        opts: Dict[str, Any] = dict(self.options)
        reader: Any = spark.read.format(fmt)

        if fmt == "csv":
            reader = reader.option("header", opts.pop("header", "true"))
            reader = reader.option("inferSchema", opts.pop("inferSchema", "false"))
            if self.path.lower().endswith(".tsv"):
                reader = reader.option("sep", opts.pop("sep", "\t"))

        for k, v in opts.items():
            reader = reader.option(k, v)
        return reader.load(self.path)

    def read(self) -> "DataFrame":
        """Alias for :meth:`dataframe` matching the foundry transforms API.

        Example:
            >>> df = Input("abfss://.../path/to/table/").read()
        """
        return self.dataframe()


# ---------- Output ----------------------------------------------------------

@dataclass(init=False)
class Output:
    """A sink. Call `.write(df)` or use a `@transform_df` decorator.

    Examples:
        Write a DataFrame to a managed Delta table (default):

        >>> out = Output.table("db.table")
        >>> out.write(df)

        Write to abfss as Parquet, partitioned:

        >>> Output(
        ...     "abfss://container@acct.../path/to/table/",
        ...     format="parquet",
        ...     mode="overwrite",
        ...     partition_by="group",
        ... ).write(df)

        Upsert via Delta MERGE on a managed table:

        >>> Output.table("db.table", mode="merge", merge_on="id").write(df)

        Write multiple sheets to one .xlsx file:

        >>> Output("abfss://.../report.xlsx").write({"a": df1, "b": df2})
    """

    path: PathLike
    format: Optional[str] = None
    mode: str = "overwrite"
    partition_by: Optional[PartitionLike] = None
    table_format: str = "delta"
    merge_on: Optional[PartitionLike] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        path: PathLike,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_by: Optional[PartitionLike] = None,
        table_format: str = "delta",
        merge_on: Optional[PartitionLike] = None,
        **options: Any,
    ) -> None:
        self.path = path
        self.format = format
        self.mode = mode
        self.partition_by = partition_by
        self.table_format = table_format
        self.merge_on = merge_on
        self.options = dict(options)

    @classmethod
    def table(
        cls,
        name: str,
        *,
        mode: str = "overwrite",
        format: str = "delta",
        partition_by: Optional[PartitionLike] = None,
        merge_on: Optional[PartitionLike] = None,
        **options: Any,
    ) -> "Output":
        """Construct an :class:`Output` that writes to a registered managed table.

        Examples:
            >>> Output.table("db.table")
            >>> Output.table("db.table", mode="append")
            >>> Output.table("db.table", mode="merge", merge_on=["id"])
            >>> Output.table("db.table", partition_by="group")
        """
        return cls(
            path=name,
            format="table",
            mode=mode,
            partition_by=partition_by,
            table_format=format,
            merge_on=merge_on,
            **options,
        )

    @property
    def fmt(self) -> str:
        """Resolve the output format without remote I/O.

        Outputs may not exist yet, so an unmarked `abfss://` path defaults to
        delta instead of trying to inspect the destination.

        Examples:
            >>> Output("out.csv").fmt
            'csv'
            >>> Output("abfss://.../folder/").fmt
            'delta'
            >>> Output.table("db.table").fmt
            'table'
        """
        return self.format or _infer_format(self.path) or "delta"

    def write(self, df: WriteData) -> None:
        """Write ``df`` to this sink.

        Routes to the right code path based on ``self.mode`` and ``self.fmt``:

        - ``mode="merge"`` → :meth:`merge_into` using ``merge_on``.
        - ``fmt == "excel"`` → driver-side xlsx via pandas.
        - ``fmt == "table"`` → ``saveAsTable`` with ``table_format``.
        - otherwise → ``df.write.format(fmt).mode(mode).save(path)``.

        Examples:
            Overwrite a managed Delta table:

            >>> Output.table("db.table").write(df)

            Append to an abfss Parquet folder:

            >>> Output("abfss://.../path/to/table/", format="parquet", mode="append").write(df)

            Merge upsert by key:

            >>> Output.table("db.table", mode="merge", merge_on="id").write(df)

            Multi-sheet Excel report:

            >>> Output("abfss://.../report.xlsx").write({"a": df1, "b": df2})
        """
        fmt: str = self.fmt

        if self.mode == "merge":
            if self.merge_on is None:
                raise ValueError("mode='merge' requires merge_on=")
            if isinstance(df, dict):
                raise TypeError("merge does not support Dict[str, DataFrame]")
            self.merge_into(df, on=self.merge_on)
            return

        if fmt == "excel":
            _write_excel(df, self.path, **self.options)
            return

        if isinstance(df, dict):
            raise TypeError(
                f"Dict[str, DataFrame] is only supported for excel outputs; "
                f"got format={fmt!r}"
            )

        opts: Dict[str, Any] = dict(self.options)

        if fmt == "table":
            writer: Any = df.write.format(self.table_format).mode(self.mode)
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

    def write_dataframe(self, df: WriteData) -> None:
        """Alias for :meth:`write` matching the foundry transforms API.

        Example:
            >>> Output.table("db.table").write_dataframe(df)
        """
        self.write(df)

    def merge_into(
        self,
        df: "DataFrame",
        on: PartitionLike,
        when_matched_update: bool = True,
        when_not_matched_insert: bool = True,
    ) -> None:
        """Delta MERGE upsert by key(s). Target must already exist.

        Examples:
            Single-key upsert:

            >>> Output.table("db.table").merge_into(df, on="id")

            Composite key:

            >>> Output.table("db.table").merge_into(df, on=["id", "fk_id"])

            Insert-only (no updates on existing rows):

            >>> Output.table("db.audit").merge_into(
            ...     df, on="id", when_matched_update=False,
            ... )

            Path-based target:

            >>> Output("abfss://.../path/to/table/").merge_into(df, on="id")
        """
        from delta.tables import DeltaTable

        spark = get_spark()
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
    """Read xlsx through pandas, then lift the result into Spark."""
    import pandas as pd

    if _is_abfss(path):
        local: str = _local_tmp(path)
        _nbutils().fs.cp(path, "file://" + local, recurse=False)
        pdf = pd.read_excel(local, sheet_name=sheet_name, **kwargs)
    else:
        pdf = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    return get_spark().createDataFrame(pdf)


def _write_excel(
    data: WriteData,
    path: PathLike,
    sheet_name: str = "Sheet1",
    **kwargs: Any,
) -> None:
    """Write one or many DataFrames to a single .xlsx file.

    Excel is intentionally driver-side. This is useful for report-sized data,
    not large tables.
    """
    import pandas as pd

    if isinstance(data, dict):
        sheets: Dict[str, Any] = {n: df.toPandas() for n, df in data.items()}
    else:
        sheets = {sheet_name: data.toPandas()}

    target: str = _local_tmp(path) if _is_abfss(path) else path
    with pd.ExcelWriter(target) as writer:
        for name, pdf in sheets.items():
            pdf.to_excel(writer, sheet_name=name, index=False, **kwargs)
    if _is_abfss(path):
        _nbutils().fs.cp("file://" + target, path, recurse=False)


# ---------- decorators ------------------------------------------------------

TransformDFFn = Callable[..., WriteData]
TransformFn = Callable[..., Any]
RunDFFn = Callable[[], WriteData]
RunFn = Callable[[], Any]


def transform_df(
    *,
    output: Output,
    **inputs: Input,
) -> Callable[[TransformDFFn], RunDFFn]:
    """Decorator: function returns data, the wrapper writes it to ``output``.

    The decorated function receives one DataFrame per ``inputs`` keyword
    argument. Its return value is written via ``output.write``. Calling the
    wrapper executes the whole transform.

    Examples:
        Single input, single output:

        >>> @transform_df(
        ...     output=Output.table("db.output"),
        ...     source=Input.table("db.input"),
        ... )
        ... def clean(source):
        ...     return source.where("value > 0")
        >>> clean()  # reads, transforms, writes

        Multi-input join:

        >>> @transform_df(
        ...     output=Output.table("db.output"),
        ...     left=Input.table("db.left"),
        ...     right=Input.table("db.right"),
        ... )
        ... def enrich(left, right):
        ...     return left.join(right, on="fk_id")

        Multi-sheet Excel report:

        >>> @transform_df(
        ...     output=Output("abfss://.../report.xlsx"),
        ...     source=Input.table("db.input"),
        ... )
        ... def report(source):
        ...     return {"data": source, "by_group": source.groupBy("group").count()}
    """
    _validate(output, inputs)

    def deco(fn: TransformDFFn) -> RunDFFn:
        @wraps(fn)
        def run() -> WriteData:
            kwargs: Dict[str, "DataFrame"] = {
                name: inp.dataframe() for name, inp in inputs.items()
            }
            result: Optional[WriteData] = fn(**kwargs)
            if result is None:
                raise ValueError(
                    f"{fn.__name__} returned None; transform_df expects a "
                    "DataFrame or Dict[str, DataFrame]."
                )
            output.write(result)
            return result

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
    """Lower-level decorator that passes raw :class:`Input`/:class:`Output` objects.

    Use this when the function needs to drive reads and writes itself — e.g.
    streaming, conditional sinks, or multiple writes from one body. Unlike
    :func:`transform_df`, the decorator does not call ``write`` for you.

    Examples:
        Single output, manual write:

        >>> @transform(
        ...     output=Output.table("db.output"),
        ...     source=Input.table("db.input"),
        ... )
        ... def go(source, output):
        ...     output.write(source.dataframe().where("value > 0"))
        >>> go()

        Multiple outputs:

        >>> @transform(
        ...     outputs={
        ...         "matched": Output.table("db.matched"),
        ...         "unmatched": Output.table("db.unmatched"),
        ...     },
        ...     source=Input.table("db.input"),
        ... )
        ... def split(source, matched, unmatched):
        ...     df = source.dataframe()
        ...     matched.write(df.where("value > 0"))
        ...     unmatched.write(df.where("value <= 0"))
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
