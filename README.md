# spark_lib

Small `transforms.api`-style helpers for Azure Synapse Spark notebooks.
Made by Artin.

The library keeps the original notebook-friendly API while also working as an
importable Python package:

```python
from spark_lib import Input, Output, transform_df
from spark_lib import clean_columns, dedupe, quiet_azure_logging
```

Synapse usually provides an active `spark` session. If your runtime does not,
register it once:

```python
import spark_lib

spark_lib.set_spark(spark)
```

## What Is Included

| Module | Purpose |
| --- | --- |
| `spark_lib.transforms` | `Input`/`Output` wrappers, `@transform_df` and `@transform` decorators. |
| `spark_lib.cleanup` | Column normalization, deduplication, parallel job runner, log helpers. |
| `spark_lib.matching` | Keyword search and fuzzy/ML matching for filling missing keys. |
| `spark_lib.delta` | Delta primitives — version lookup, snapshot merge, CDF merge. |
| `spark_lib.sync` | Incremental Delta-to-managed-table sync orchestration. |
| `spark_lib.session` | `get_spark` / `set_spark` for runtime SparkSession lookup. |

---

## `spark_lib.transforms` — Inputs, Outputs, Decorators

### `Input`

A lazy reference to a dataset. Format is inferred from the path extension; bare
`abfss://` folders are peeked once for `_delta_log` to distinguish delta from
parquet/csv-style folders.

```python
Input("abfss://raw@acct.dfs.core.windows.net/orders/")            # delta
Input("abfss://raw@acct.dfs.core.windows.net/orders.csv")         # csv
Input("abfss://raw@acct.dfs.core.windows.net/orders.xlsx")        # excel
Input.table("sales.daily_orders")                                 # managed table
```

Pass any reader options as kwargs:

```python
Input("abfss://.../sales.csv", header="true", inferSchema="false", sep=",")
```

CSV reads default to `header="true"` and `inferSchema="false"` (set explicitly
to opt into the double-pass schema inference).

### `Output`

A sink. Resolves the same way `Input` does. Outputs may not exist yet, so
unmarked `abfss://` outputs default to delta without peeking.

```python
Output("abfss://lab@acct.dfs.core.windows.net/sales/")            # delta path
Output.table("sales.daily_orders")                                # managed delta
Output.table(
    "sales.upserts",
    mode="merge",
    merge_on=["order_id"],
)                                                                  # delta upsert
```

The `mode="merge"` path requires `merge_on=`. It calls `Output.merge_into`
under the hood.

### `@transform_df` / `@transform`

```python
from spark_lib import Input, Output, clean_columns, dedupe, transform_df


@transform_df(
    output=Output.table("sales.daily_orders", mode="overwrite"),
    orders=Input("abfss://raw@acct.dfs.core.windows.net/orders/"),
    customers=Input("abfss://raw@acct.dfs.core.windows.net/customers.csv"),
)
def daily_orders(orders, customers):
    joined = orders.join(customers, "customer_id")
    cleaned = clean_columns(joined)
    return dedupe(cleaned, pks=["order_id"], order_by="updated_at")


daily_orders()
```

`transform_df` is for one DataFrame in → one DataFrame out → write. Use the
lower-level `@transform` when you need raw `Input`/`Output` objects or
multiple sinks.

---

## `spark_lib.cleanup` — Normalization, Dedup, Parallelism

### `clean_columns(df)`

Renames every column to snake_case ASCII. Strips accents, collapses runs of
non-alphanumeric to `_`, prefixes leading digits with `_`, and disambiguates
collisions with `_0`, `_1`, …

### `dedupe(df, pks, order_by, descending=True)`

Keeps one row per primary-key group, picked by `order_by`. Window-based, so
no shuffle beyond the partitionBy.

### `run_parallel(fn, jobs, max_workers=4, pool=None, fail_fast=False)`

Runs `fn(**job)` for every job concurrently via `ThreadPoolExecutor`. If
`pool` is given, sets `spark.scheduler.pool` on the worker thread (requires
Spark FAIR scheduling). On `fail_fast=True`, the first failure cancels
in-flight futures. Otherwise, exceptions are returned in-place inside the
results list and a single warning per failure is logged (full tracebacks
go to debug level).

### `drop_database_tables(database, *, tables=None, include_views=False, max_workers=8, pool=None, dry_run=False)`

Drops all managed/external tables in a database concurrently. Views are skipped
unless `include_views=True`.

```python
from spark_lib import drop_database_tables

results = drop_database_tables("lab", max_workers=8)
failures = [r for r in results if isinstance(r, BaseException)]
if failures:
    raise RuntimeError(f"{len(failures)} drops failed")
```

### `quiet_azure_logging(level=WARNING)`

Raises noisy Azure / py4j / msal / urllib3 loggers above normal notebook
chatter.

---

## `spark_lib.matching` — Keyword Search and Fuzzy Joins

```python
from spark_lib import (
    fill_missing_from_match,
    fuzzy_match,
    infer_key_from_text,
    ml_fuzzy_match,
    search_database,
)

hits = search_database("sales", "acme", limit_tables=20)

matches = fuzzy_match(
    left=orders,
    right=customers,
    left_on="customer_name",
    right_on="legal_name",
    block_on="country",
    threshold=0.82,
)

filled = fill_missing_from_match(
    left=orders,
    right=customers,
    left_on="customer_name",
    right_on="legal_name",
    fill_cols=["customer_id", "billing_city"],
    block_on="country",
    threshold=0.82,
)

resolved = infer_key_from_text(
    left=missing_ids,
    reference=known_businesses,
    key_col="business_id",
    text_col="business_name",
    method="ml",
    threshold=0.84,
)
```

- `fuzzy_match` uses Levenshtein scoring. Without `block_on`, it joins only
  rows sharing the first two normalized characters to avoid a cross join.
- `ml_fuzzy_match` and `infer_key_from_text(method="ml")` use PySpark ML
  `HashingTF` plus `MinHashLSH` over normalized character n-grams. Better
  for business names ("Wal Mart" vs "Walmart").
- All matchers emit audit columns: matched right-side row, score, and
  distance — auditable, never silently trusted.

---

## `spark_lib.delta` — Delta Lake Primitives

Generic helpers used by `spark_lib.sync` and reusable on their own.

### `current_delta_version(path) -> Optional[int]`

Cheap, file-only lookup. Reads `_delta_log/_last_checkpoint` first; falls back
to listing the log directory and picking the max numeric `.json` commit.
Returns `None` when neither is readable so callers can decide whether to keep
going.

```python
from spark_lib.delta import current_delta_version

v = current_delta_version("abfss://.../orders/")  # 137 or None
```

This intentionally avoids `DeltaTable.forPath` (URI-strict) and
`DESCRIBE HISTORY` (full log scan).

### `snapshot_merge(target_table, df, on, *, delete_unmatched=True)`

Wraps the `forName + merge + whenMatchedUpdate + whenNotMatchedInsert +
whenNotMatchedBySourceDelete` chain into one call. The target must already
exist.

```python
from spark_lib.delta import snapshot_merge

snapshot_merge("lab.orders", new_snapshot, on=["order_id"])
```

Set `delete_unmatched=False` for an upsert that does not delete rows missing
from the source.

### `cdf_merge(target_table, cdf, on, *, metadata_cols=...) -> bool`

Applies a Delta CDF DataFrame onto `target_table`. Filters
`update_preimage`, dedupes by latest `commit_version`/`commit_timestamp` per
key, and applies inserts / updates / deletes. Returns `True` when a merge
happened, `False` for an empty CDF window.

```python
from spark_lib.delta import cdf_merge, read_cdf

cdf = read_cdf("abfss://.../orders/", start_version=last + 1)
applied = cdf_merge("lab.orders", cdf, on=["order_id"])
```

### `read_cdf(src_path, start_version) -> DataFrame`

Convenience wrapper around `spark.read.format("delta").option("readChangeFeed",
"true").option("startingVersion", N).load(...)`.

### `merge_condition(pks)` and `column_map(cols)`

```python
merge_condition(["order_id", "tenant_id"])
# -> "t.order_id = s.order_id AND t.tenant_id = s.tenant_id"

column_map(["order_id", "amount"])
# -> {"order_id": "s.order_id", "amount": "s.amount"}
```

---

## `spark_lib.sync` — Incremental Delta Sync

End-to-end orchestration: snapshot vs CDF decision, fallback, state tracking,
parallel execution.

### Quick Start

```python
from spark_lib.sync import SyncSpec, SyncState, run_sync

specs: list[SyncSpec] = [
    {
        "src_key":   "sales.orders",
        "src_path":  "abfss://raw@acct.dfs.core.windows.net/SALES/ORDERS/1.2/",
        "dst_table": "lab.orders",
        "pks":       ["order_id"],
    },
    # ... more specs
]

state = SyncState("lab.__spark_lib_delta_sync_state")
successes, failures = run_sync(specs, state, max_workers=8, pool="delta_sync")
```

### `SyncSpec`

A `TypedDict` describing one source:

| Key | Meaning |
| --- | --- |
| `src_key` | Stable identifier, e.g. `"db.table"`. Used as the state-table primary key. |
| `src_path` | Source Delta folder URI. |
| `dst_table` | Managed Delta table name. |
| `pks` | Primary key columns for merge predicates and CDF dedup. |

### `SyncState`

Persists last-synced Delta version per `src_key` in a managed Delta table.

| Method | Purpose |
| --- | --- |
| `ensure()` | Create the state table if it does not exist. |
| `get(src_key)` | Last synced version for one source, or `None`. |
| `load_all()` | Whole state table as `Dict[src_key, version]` — one scan. |
| `upsert(result)` | Write one result row. |
| `upsert_all(results)` | Batch many result rows in one MERGE. |

### `sync_delta_to_table(...)`

Sync one source. The decision tree:

1. No prior version, or destination missing → **initial snapshot** (full overwrite).
2. Source version unreadable → **snapshot merge** (preserves time travel).
3. `last_version >= current_version` → **already current** (no work).
4. Otherwise → **CDF merge** from `last_version + 1` to `current_version`.
   On any CDF error (e.g. CDF not enabled on source), falls back to
   snapshot merge automatically.

### `run_sync(specs, state, *, max_workers=4, pool=None) -> (successes, failures)`

Parallel orchestrator. Optimizations:

- **Single state-table scan up front** (`state.load_all()`), threaded into
  workers — avoids N small queries against the same tiny table.
- **Single batched state MERGE** at the end (`state.upsert_all`) — one Delta
  transaction instead of one per success.
- State writes are sequential by design (avoids concurrent Delta-transaction
  conflicts on the state table).

Returns a tuple `(List[SyncResult], List[BaseException])`. State is written
only for successes; failures bubble up so you can inspect or re-raise.

---

## `spark_lib.session` — SparkSession Lookup

```python
from spark_lib import get_spark, set_spark
```

`get_spark()` returns the registered session, or Spark's active session, or
raises. The package never calls `SparkSession.builder.getOrCreate()` so it
won't mutate Synapse's pre-injected session.

`set_spark(spark)` is for local scripts/tests where no session is active yet.

---

## Compatibility Imports

Existing notebook code can still import the old module names:

```python
from cleanup import clean_columns, dedupe
from synapse_transforms import Input, Output, transform, transform_df
```

Those files re-export the package implementation.

---

## Install

For local development:

```bash
python -m pip install -e .
```

For Excel support, install the optional dependencies:

```bash
python -m pip install -e ".[excel]"
```

For a local dev environment with build tooling, Spark, and Excel dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Build A Wheel

```bash
python -m pip install --upgrade build
python -m build --wheel
```

The wheel will be written to `dist/`, for example:

```text
dist/spark_lib-0.1.0-py3-none-any.whl
```

## Synapse Bootstrap

When Synapse workspace imports are inconvenient, copy `synapse_bootstrap.py`
and the built wheel to ADLS, then load the bootstrap file in a notebook:

```python
code = mssparkutils.fs.head(
    "abfss://<container>@<account>.dfs.core.windows.net/libs/synapse_bootstrap.py",
    200000,
)
exec(code)

install_wheel_from_abfss(
    "abfss://<container>@<account>.dfs.core.windows.net/libs/spark_lib-0.1.0-py3-none-any.whl"
)
```

For a one-off single-file load, the bootstrap also exposes:

```python
exec_py_from_abfss(
    "abfss://<container>@<account>.dfs.core.windows.net/libs/some_file.py",
    globals(),
)
```

---

## Conventions

- Paths are always `abfss://` (or managed-table names). No `s3://`, `dbfs:`,
  or `file://` code paths.
- Type hints come from `typing` (`Dict`, `List`, `Optional`, …) — Pylance
  flags these as deprecated; that's tolerated.
- `spark` is referenced as a notebook-injected global in user code. The
  package itself uses `get_spark()` so it never bootstraps a session.
- Bare `abfss://` input directories are peeked once to distinguish delta
  from parquet/csv-style folders.
- Output paths are not peeked because they may not exist yet; unmarked
  `abfss://` outputs default to delta.
- Excel I/O is driver-side via pandas — for report-sized data, not bulk.
- `run_parallel` uses threads and assumes Spark FAIR scheduling is configured
  at session start when using scheduler pools.

## License

MIT License. Copyright (c) 2026 Artin.
