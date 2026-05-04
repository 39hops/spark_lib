"""Keyword search and fuzzy matching helpers for Spark DataFrames."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from .session import get_spark

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


NameList = Optional[Union[str, Iterable[str]]]
_LEFT_KEY = "__spark_lib_left_id"
_RIGHT_KEY = "__spark_lib_right_id"
_LEFT_VALUE = "__spark_lib_left_value"
_RIGHT_VALUE = "__spark_lib_right_value"
_LEFT_NORM = "__spark_lib_left_norm"
_RIGHT_NORM = "__spark_lib_right_norm"
_MATCH_RIGHT_KEY = "__spark_lib_match_right_id"
_TOKENS = "__spark_lib_tokens"
_COMPACT = "__spark_lib_compact"
_FEATURES = "__spark_lib_features"
_HASHES = "__spark_lib_hashes"
_ACCENT_CHARS = "àáâãäåāăąçćčďèéêëēėęěìíîïīįłñńòóôõöøōřśšťùúûüūůýÿžźż"
_ASCII_CHARS = "aaaaaaaaacccdeeeeeeeeiiiiiilnnooooooorsstuuuuuuyyzzz"


def normalize_text(col: Union[str, "Column"]) -> "Column":
    """Return a Spark Column normalized for matching."""
    from pyspark.sql import functions as F

    expr: "Column" = F.col(col) if isinstance(col, str) else col
    expr = F.lower(F.coalesce(expr.cast("string"), F.lit("")))
    expr = F.translate(expr, _ACCENT_CHARS, _ASCII_CHARS)
    expr = F.regexp_replace(expr, r"[^a-z0-9]+", " ")
    expr = F.regexp_replace(expr, r"\s+", " ")
    return F.trim(expr)


def search_database(
    database: str,
    keyword: str,
    *,
    tables: Optional[Iterable[str]] = None,
    columns: Optional[Iterable[str]] = None,
    case_sensitive: bool = False,
    sample_rows: int = 1,
    limit_tables: Optional[int] = None,
) -> "DataFrame":
    """Search string columns in a database and return match counts/samples."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    spark = get_spark()
    if keyword == "":
        raise ValueError("keyword must not be empty")
    if sample_rows < 0:
        raise ValueError("sample_rows must be >= 0")
    if limit_tables is not None and limit_tables < 0:
        raise ValueError("limit_tables must be >= 0")

    wanted_cols = set(columns or [])
    names: List[str]
    if tables is None:
        names = [t.name for t in spark.catalog.listTables(database)]
    else:
        names = list(tables)
    if limit_tables is not None:
        names = names[:limit_tables]

    rows: List[Tuple[str, str, int, List[str]]] = []
    needle = keyword if case_sensitive else keyword.lower()

    for table in names:
        full_name = table if "." in table else f"{database}.{table}"
        df = spark.table(full_name)
        for field in df.schema.fields:
            if not isinstance(field.dataType, StringType):
                continue
            if wanted_cols and field.name not in wanted_cols:
                continue

            value = F.col(field.name)
            haystack = value if case_sensitive else F.lower(value)
            matches = df.where(haystack.contains(needle))
            count = matches.count()
            if count == 0:
                continue

            samples: List[str] = []
            if sample_rows > 0:
                samples = [
                    r["sample"]
                    for r in (
                        matches.select(value.alias("sample"))
                               .limit(sample_rows)
                               .collect()
                    )
                    if r["sample"] is not None
                ]
            rows.append((full_name, field.name, count, samples))

    schema = StructType(
        [
            StructField("table_name", StringType(), False),
            StructField("column_name", StringType(), False),
            StructField("match_count", LongType(), False),
            StructField("sample_values", ArrayType(StringType()), False),
        ]
    )
    return spark.createDataFrame(rows, schema)


def fuzzy_match(
    left: "DataFrame",
    right: "DataFrame",
    *,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: NameList = None,
    threshold: float = 0.8,
    max_distance: Optional[int] = None,
    min_chars: int = 2,
    keep_all_candidates: bool = False,
) -> "DataFrame":
    """Return fuzzy matches between two DataFrames, ranked per left row."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    _validate_threshold(threshold)
    if min_chars < 1:
        raise ValueError("min_chars must be >= 1")
    left_key = left_id or _LEFT_KEY
    right_key = right_id or _RIGHT_KEY
    blocks = _as_list(block_on)
    _require_columns(
        left,
        [left_on] + blocks + ([left_id] if left_id else []),
        "left",
    )
    _require_columns(
        right, [right_on] + blocks + ([right_id] if right_id else []), "right"
    )
    if left_id is None:
        _require_absent(left, [_LEFT_KEY], "left")
    if right_id is None:
        _require_absent(right, [_RIGHT_KEY], "right")
    _require_absent(left, [_LEFT_VALUE, _LEFT_NORM], "left")
    _require_absent(right, [_RIGHT_VALUE, _RIGHT_NORM], "right")

    ldf = _with_key(left, left_key, left_id is None)
    rdf = _with_key(right, right_key, right_id is None)
    ldf = (
        ldf.withColumn(_LEFT_VALUE, F.col(left_on).cast("string"))
           .withColumn(_LEFT_NORM, normalize_text(_LEFT_VALUE))
    )
    rdf = (
        rdf.withColumn(_RIGHT_VALUE, F.col(right_on).cast("string"))
           .withColumn(_RIGHT_NORM, normalize_text(_RIGHT_VALUE))
    )
    ldf = ldf.where(F.length(F.col(_LEFT_NORM)) >= F.lit(min_chars))
    rdf = rdf.where(F.length(F.col(_RIGHT_NORM)) >= F.lit(min_chars))

    l = ldf.alias("l")
    r = rdf.alias("r")
    cond = _join_condition(blocks)
    joined = l.join(r, cond, "inner")

    distance = F.levenshtein(F.col(f"l.{_LEFT_NORM}"), F.col(f"r.{_RIGHT_NORM}"))
    max_len = F.greatest(
        F.length(F.col(f"l.{_LEFT_NORM}")),
        F.length(F.col(f"r.{_RIGHT_NORM}")),
    )
    score = F.when(max_len == 0, F.lit(0.0)).otherwise(
        1.0 - (distance.cast("double") / max_len.cast("double"))
    )

    matched = joined.select(
        F.col(f"l.{left_key}").alias("left_id"),
        F.col(f"r.{right_key}").alias("right_id"),
        F.col(f"l.{_LEFT_VALUE}").alias("left_value"),
        F.col(f"r.{_RIGHT_VALUE}").alias("right_value"),
        F.col(f"l.{_LEFT_NORM}").alias("left_norm"),
        F.col(f"r.{_RIGHT_NORM}").alias("right_norm"),
        distance.alias("match_distance"),
        score.alias("match_score"),
    )
    matched = matched.where(F.col("match_score") >= F.lit(float(threshold)))
    if max_distance is not None:
        matched = matched.where(F.col("match_distance") <= F.lit(int(max_distance)))

    window = Window.partitionBy("left_id").orderBy(
        F.desc("match_score"), F.asc("match_distance"), F.asc("right_id")
    )
    ranked = matched.withColumn("match_rank", F.row_number().over(window))
    if keep_all_candidates:
        return ranked
    return ranked.where(F.col("match_rank") == 1)


def fill_missing_from_match(
    left: "DataFrame",
    right: "DataFrame",
    *,
    left_on: str,
    right_on: str,
    fill_cols: Iterable[str],
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: NameList = None,
    threshold: float = 0.8,
    max_distance: Optional[int] = None,
    min_chars: int = 2,
    overwrite: bool = False,
    audit_prefix: str = "_match",
) -> "DataFrame":
    """Fill selected left columns from the best fuzzy right-side match."""
    from pyspark.sql import functions as F

    cols = _unique(list(fill_cols))
    if not cols:
        raise ValueError("fill_cols must not be empty")
    fill_set = set(cols)
    left_key = left_id or _LEFT_KEY
    right_key = right_id or _RIGHT_KEY
    blocks = _as_list(block_on)
    audit_cols = [
        f"{audit_prefix}_right_id",
        f"{audit_prefix}_right_text",
        f"{audit_prefix}_score",
        f"{audit_prefix}_distance",
    ]
    _require_columns(
        left,
        [left_on] + cols + blocks + ([left_id] if left_id else []),
        "left",
    )
    _require_columns(
        right, [right_on] + cols + blocks + ([right_id] if right_id else []), "right"
    )
    _require_absent(left, audit_cols, "left")
    if left_id is None:
        _require_absent(left, [_LEFT_KEY], "left")
    if right_id is None:
        _require_absent(right, [_RIGHT_KEY], "right")
    _require_absent(left, [_MATCH_RIGHT_KEY], "left")
    ldf = _with_key(left, left_key, left_id is None)
    rdf = _with_key(right, right_key, right_id is None)

    matches = fuzzy_match(
        ldf,
        rdf,
        left_on=left_on,
        right_on=right_on,
        left_id=left_key,
        right_id=right_key,
        block_on=block_on,
        threshold=threshold,
        max_distance=max_distance,
        min_chars=min_chars,
    ).select(
        F.col("left_id").alias(left_key),
        F.col("right_id").alias(_MATCH_RIGHT_KEY),
        F.col("right_value").alias(f"{audit_prefix}_right_text"),
        F.col("match_score").alias(f"{audit_prefix}_score"),
        F.col("match_distance").alias(f"{audit_prefix}_distance"),
    )

    right_cols = _unique([right_key] + cols)
    right_lookup = rdf.select(
        *[F.col(c).alias(f"__right_{c}") for c in right_cols]
    )
    joined = (
        ldf.join(matches, left_key, "left")
           .join(
               right_lookup,
               F.col(_MATCH_RIGHT_KEY) == F.col(f"__right_{right_key}"),
               "left",
           )
    )

    selected: List["Column"] = []
    for name in left.columns:
        if name in fill_set:
            replacement = F.col(f"__right_{name}")
            value = (
                replacement if overwrite else F.coalesce(F.col(name), replacement)
            )
            selected.append(value.alias(name))
        else:
            selected.append(F.col(name))

    selected.extend(
        [
            F.col(_MATCH_RIGHT_KEY).alias(f"{audit_prefix}_right_id"),
            F.col(f"{audit_prefix}_right_text"),
            F.col(f"{audit_prefix}_score"),
            F.col(f"{audit_prefix}_distance"),
        ]
    )
    return joined.select(*selected)


def ml_fuzzy_match(
    left: "DataFrame",
    right: "DataFrame",
    *,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: NameList = None,
    threshold: float = 0.8,
    min_chars: int = 2,
    ngram_size: int = 3,
    num_features: int = 1 << 18,
    num_hash_tables: int = 3,
    keep_all_candidates: bool = False,
) -> "DataFrame":
    """Return fuzzy matches using PySpark ML MinHashLSH over char n-grams."""
    from pyspark.ml.feature import HashingTF, MinHashLSH
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    _validate_threshold(threshold)
    if min_chars < 1:
        raise ValueError("min_chars must be >= 1")
    if ngram_size < 1:
        raise ValueError("ngram_size must be >= 1")
    if num_features <= 0:
        raise ValueError("num_features must be > 0")
    if num_hash_tables <= 0:
        raise ValueError("num_hash_tables must be > 0")

    left_key = left_id or _LEFT_KEY
    right_key = right_id or _RIGHT_KEY
    blocks = _as_list(block_on)
    _require_columns(
        left,
        [left_on] + blocks + ([left_id] if left_id else []),
        "left",
    )
    _require_columns(
        right, [right_on] + blocks + ([right_id] if right_id else []), "right"
    )
    if left_id is None:
        _require_absent(left, [_LEFT_KEY], "left")
    if right_id is None:
        _require_absent(right, [_RIGHT_KEY], "right")
    reserved = [
        _LEFT_VALUE,
        _RIGHT_VALUE,
        _LEFT_NORM,
        _RIGHT_NORM,
        _TOKENS,
        _COMPACT,
        _FEATURES,
    ]
    _require_absent(left, reserved, "left")
    _require_absent(right, reserved, "right")

    ldf = (
        _with_key(left, left_key, left_id is None)
        .withColumn(_LEFT_VALUE, F.col(left_on).cast("string"))
        .withColumn(_LEFT_NORM, normalize_text(_LEFT_VALUE))
        .withColumn(_COMPACT, F.regexp_replace(F.col(_LEFT_NORM), " ", ""))
        .withColumn(_TOKENS, _char_ngrams(_COMPACT, ngram_size))
        .where(F.length(F.col(_LEFT_NORM)) >= F.lit(min_chars))
    )
    rdf = (
        _with_key(right, right_key, right_id is None)
        .withColumn(_RIGHT_VALUE, F.col(right_on).cast("string"))
        .withColumn(_RIGHT_NORM, normalize_text(_RIGHT_VALUE))
        .withColumn(_COMPACT, F.regexp_replace(F.col(_RIGHT_NORM), " ", ""))
        .withColumn(_TOKENS, _char_ngrams(_COMPACT, ngram_size))
        .where(F.length(F.col(_RIGHT_NORM)) >= F.lit(min_chars))
    )

    hashing = HashingTF(
        inputCol=_TOKENS,
        outputCol=_FEATURES,
        numFeatures=num_features,
        binary=True,
    )
    lfeat = hashing.transform(ldf)
    rfeat = hashing.transform(rdf)
    lfeat = lfeat.where(F.size(F.col(_TOKENS)) > 0)
    rfeat = rfeat.where(F.size(F.col(_TOKENS)) > 0)

    fit_data = lfeat.select(_FEATURES).unionByName(rfeat.select(_FEATURES))
    model = MinHashLSH(
        inputCol=_FEATURES,
        outputCol=_HASHES,
        numHashTables=num_hash_tables,
    ).fit(fit_data)
    max_jaccard_distance = float(1.0 - threshold)
    joined = model.approxSimilarityJoin(
        lfeat,
        rfeat,
        max_jaccard_distance,
        distCol="match_distance",
    )

    if blocks:
        joined = joined.where(_ml_block_condition(blocks))

    matched = joined.select(
        F.col(f"datasetA.{left_key}").alias("left_id"),
        F.col(f"datasetB.{right_key}").alias("right_id"),
        F.col(f"datasetA.{_LEFT_VALUE}").alias("left_value"),
        F.col(f"datasetB.{_RIGHT_VALUE}").alias("right_value"),
        F.col(f"datasetA.{_LEFT_NORM}").alias("left_norm"),
        F.col(f"datasetB.{_RIGHT_NORM}").alias("right_norm"),
        F.col("match_distance"),
        (1.0 - F.col("match_distance")).alias("match_score"),
    )
    window = Window.partitionBy("left_id").orderBy(
        F.desc("match_score"), F.asc("match_distance"), F.asc("right_id")
    )
    ranked = matched.withColumn("match_rank", F.row_number().over(window))
    if keep_all_candidates:
        return ranked
    return ranked.where(F.col("match_rank") == 1)


def infer_key_from_text(
    left: "DataFrame",
    reference: "DataFrame",
    *,
    key_col: str,
    text_col: str,
    reference_key_col: Optional[str] = None,
    reference_text_col: Optional[str] = None,
    left_id: Optional[str] = None,
    block_on: NameList = None,
    method: str = "ml",
    threshold: float = 0.8,
    max_distance: Optional[int] = None,
    min_chars: int = 2,
    ngram_size: int = 3,
    num_features: int = 1 << 18,
    num_hash_tables: int = 3,
    overwrite: bool = False,
    audit_prefix: str = "_match",
) -> "DataFrame":
    """Infer a missing key from a fuzzy match on a text column.

    `reference` should contain the trusted key/text pairs, such as
    `(business_id, business_name)`. `left` may have a null key column or no
    key column at all.
    """
    from pyspark.sql import functions as F

    ref_key = reference_key_col or key_col
    ref_text = reference_text_col or text_col
    left_key = left_id or _LEFT_KEY
    blocks = _as_list(block_on)
    key_exists = key_col in left.columns
    audit_cols = [
        f"{audit_prefix}_inferred_{key_col}",
        f"{audit_prefix}_right_text",
        f"{audit_prefix}_score",
        f"{audit_prefix}_distance",
    ]

    _require_columns(left, [text_col] + blocks + ([left_id] if left_id else []), "left")
    _require_columns(reference, [ref_key, ref_text] + blocks, "reference")
    _require_absent(left, audit_cols, "left")
    if left_id is None:
        _require_absent(left, [_LEFT_KEY], "left")
    base = _with_key(left, left_key, left_id is None)
    work = base
    if key_exists and not overwrite:
        work = work.where(F.col(key_col).isNull())

    method_name = method.lower()
    if method_name in ("ml", "lsh", "minhash"):
        matched = ml_fuzzy_match(
            work,
            reference,
            left_on=text_col,
            right_on=ref_text,
            left_id=left_key,
            right_id=ref_key,
            block_on=block_on,
            threshold=threshold,
            min_chars=min_chars,
            ngram_size=ngram_size,
            num_features=num_features,
            num_hash_tables=num_hash_tables,
        )
    elif method_name in ("levenshtein", "edit_distance"):
        matched = fuzzy_match(
            work,
            reference,
            left_on=text_col,
            right_on=ref_text,
            left_id=left_key,
            right_id=ref_key,
            block_on=block_on,
            threshold=threshold,
            max_distance=max_distance,
            min_chars=min_chars,
        )
    else:
        raise ValueError("method must be 'ml' or 'levenshtein'")

    matches = matched.select(
        F.col("left_id").alias(left_key),
        F.col("right_id").alias(f"{audit_prefix}_inferred_{key_col}"),
        F.col("right_value").alias(f"{audit_prefix}_right_text"),
        F.col("match_score").alias(f"{audit_prefix}_score"),
        F.col("match_distance").alias(f"{audit_prefix}_distance"),
    )

    joined = base.join(matches, left_key, "left")
    inferred = F.col(f"{audit_prefix}_inferred_{key_col}")

    selected: List["Column"] = []
    for name in left.columns:
        if name == key_col:
            value = inferred if overwrite else F.coalesce(F.col(name), inferred)
            selected.append(value.alias(name))
        else:
            selected.append(F.col(name))
    if not key_exists:
        selected.append(inferred.alias(key_col))

    selected.extend(
        [
            inferred,
            F.col(f"{audit_prefix}_right_text"),
            F.col(f"{audit_prefix}_score"),
            F.col(f"{audit_prefix}_distance"),
        ]
    )
    return joined.select(*selected)


def _with_key(df: "DataFrame", key: str, create: bool) -> "DataFrame":
    if not create:
        return df
    from pyspark.sql import functions as F

    return df.withColumn(key, F.monotonically_increasing_id())


def _join_condition(blocks: List[str]) -> "Column":
    from functools import reduce
    from operator import and_
    from pyspark.sql import functions as F

    if blocks:
        return reduce(and_, [F.col(f"l.{c}") == F.col(f"r.{c}") for c in blocks])
    return F.substring(F.col(f"l.{_LEFT_NORM}"), 1, 2) == F.substring(
        F.col(f"r.{_RIGHT_NORM}"), 1, 2
    )


def _ml_block_condition(blocks: List[str]) -> "Column":
    from functools import reduce
    from operator import and_
    from pyspark.sql import functions as F

    return reduce(
        and_,
        [F.col(f"datasetA.{c}") == F.col(f"datasetB.{c}") for c in blocks],
    )


def _char_ngrams(col: str, n: int) -> "Column":
    from pyspark.sql import functions as F

    empty = F.array().cast("array<string>")
    return (
        F.when(F.length(F.col(col)) == 0, empty)
         .when(F.length(F.col(col)) <= F.lit(n), F.array(F.col(col)))
         .otherwise(
             F.expr(
                 f"transform(sequence(1, length({col}) - {n} + 1), "
                 f"i -> substring({col}, i, {n}))"
             )
         )
    )


def _as_list(value: NameList) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _validate_threshold(threshold: float) -> None:
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")


def _require_columns(
    df: "DataFrame",
    names: Iterable[Optional[str]],
    side: str,
) -> None:
    missing = [n for n in names if n is not None and n not in df.columns]
    if missing:
        raise ValueError(f"{side} DataFrame is missing columns: {missing}")


def _require_absent(df: "DataFrame", names: Iterable[str], side: str) -> None:
    present = [n for n in names if n in df.columns]
    if present:
        raise ValueError(
            f"{side} DataFrame already has reserved output columns: {present}"
        )


__all__ = [
    "fill_missing_from_match",
    "fuzzy_match",
    "infer_key_from_text",
    "ml_fuzzy_match",
    "normalize_text",
    "search_database",
]
