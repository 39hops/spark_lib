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

- `Input` and `Output` wrappers for managed tables and `abfss://` paths.
- `transform_df` for functions that return a DataFrame and write to one output.
- `transform` for lower-level functions that need raw inputs or multiple outputs.
- Format inference for delta, parquet, csv, tsv, json, orc, avro, and xlsx.
- Excel read/write support through pandas and Synapse filesystem utilities.
- `clean_columns`, `dedupe`, `quiet_azure_logging`, and `run_parallel`.
- Database keyword search and fuzzy matching helpers for filling missing fields.

## Example

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

## Matching Helpers

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

`infer_key_from_text(..., method="ml")` and `ml_fuzzy_match` use PySpark ML
`HashingTF` plus `MinHashLSH` over normalized business-name character n-grams.
The lower level `fuzzy_match` helper uses Levenshtein scoring. Pass `block_on=`
when possible so candidate generation stays small. Filled outputs include audit
columns for the matched right-side row, score, and distance.

## Compatibility Imports

Existing notebook code can still import the old module names:

```python
from cleanup import clean_columns, dedupe
from synapse_transforms import Input, Output, transform, transform_df
```

Those files now re-export the package implementation.

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

Build the wheel from the repo root:

```bash
python -m pip install --upgrade build
python -m build --wheel
```

The wheel will be written to `dist/`, for example:

```text
dist/spark_lib-0.1.0-py3-none-any.whl
```

Install the built wheel:

```bash
python -m pip install dist/spark_lib-0.1.0-py3-none-any.whl
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

## Notes

- The package never creates a SparkSession.
- Bare `abfss://` input directories are peeked once to distinguish delta from
  parquet/csv/json-style folders.
- Output paths are not peeked because they may not exist yet; unmarked
  `abfss://` outputs default to delta.
- Excel I/O is driver-side via pandas, so it is intended for report-sized
  data, not bulk datasets.
- `run_parallel` uses threads and assumes Spark FAIR scheduling is configured
  at session start when using scheduler pools.

## License

MIT License. Copyright (c) 2026 Artin.
