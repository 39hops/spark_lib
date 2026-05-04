# spark_lib

Small `transforms.api`-style helpers for Azure Synapse Spark notebooks.

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
