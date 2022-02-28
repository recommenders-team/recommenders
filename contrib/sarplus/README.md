# SARplus

Pronounced surplus as it's simply better if not best!

[![sarplus test and package](https://github.com/microsoft/recommenders/actions/workflows/sarplus.yml/badge.svg)](https://github.com/microsoft/recommenders/actions/workflows/sarplus.yml)
[![PyPI version](https://img.shields.io/pypi/v/pysarplus.svg)](https://pypi.org/project/pysarplus/)
[![Python version](https://img.shields.io/pypi/pyversions/pysarplus)](https://pypi.org/project/pysarplus/)
[![Maven Central version](https://img.shields.io/maven-central/v/com.microsoft.sarplus/sarplus_2.12)](https://search.maven.org/artifact/com.microsoft.sarplus/sarplus_2.12)
[![Maven Central version (Spark 3.2+)](https://img.shields.io/maven-central/v/com.microsoft.sarplus/sarplus-spark-3-2-plus_2.12?label=maven-central%20%28spark%203.2%2B%29)](https://search.maven.org/artifact/com.microsoft.sarplus/sarplus-spark-3-2-plus_2.12)


Simple Algorithm for Recommendation (SAR) is a neighborhood based
algorithm for personalized recommendations based on user transaction
history. SAR recommends items that are most **similar** to the ones
that the user already has an existing **affinity** for. Two items are
**similar** if the users that interacted with one item are also likely
to have interacted with the other. A user has an **affinity** to an
item if they have interacted with it in the past.

SARplus is an efficient implementation of this algorithm for Spark.

Features:

* Scalable PySpark based [implementation](python/pysarplus/SARPlus.py)
* Fast C++ based [predictions](python/src/pysarplus.cpp)
* Reduced memory consumption: similarity matrix cached in-memory once
  per worker, shared across python executors

## Benchmarks

| # Users | # Items | # Ratings | Runtime | Environment                                                                                                                                 | Dataset |
|---------|---------|-----------|---------|---------------------------------------------------------------------------------------------------------------------------------------------|---------|
| 2.5mio  | 35k     | 100mio    | 1.3h    | Databricks, 8 workers, [Azure Standard DS3 v2](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/) (4 core machines) |         |

## Top-K Recommendation Optimization

There are a couple of key optimizations:

* map item ids (e.g. strings) to a continuous set of indexes to
  optimize storage and simplify access
* convert similarity matrix to exactly the representation the C++
  component needs, thus enabling simple shared, memory mapping of the
  cache file and avoid parsing. This requires a customer formatter,
  written in Scala
* shared read-only memory mapping allows us to re-use the same memory
  from multiple python executors on the same worker node
* partition the input test users and past seen items by users,
  allowing for scale out
* perform as much of the work as possible in PySpark (way simpler)
* top-k computation
    + reverse the join by summing reverse joining the users past seen
      items with any related items
    + make sure to always just keep top-k items in-memory
    + use standard join using binary search between users past seen
      items and the related items

![Image of sarplus top-k recommendation optimization](https://recodatasets.z20.web.core.windows.net/images/sarplus_udf.svg)

## Usage

Two packages should be installed:
* [pysarplus@PyPI](https://pypi.org/project/pysarplus/)
* [sarplus@MavenCentralRepository](https://search.maven.org/artifact/com.microsoft.sarplus/sarplus_2.12) (or [sarplus-spark-3-2-plus@MavenCentralRepository](https://search.maven.org/artifact/com.microsoft.sarplus/sarplus-spark-3-2-plus_2.12) if run on Spark 3.2+)


### Python

```python
from pysarplus import SARPlus

# spark dataframe with user/item/rating/optional timestamp tuples
train_df = spark.createDataFrame(
    [(1, 1, 1), (1, 2, 1), (2, 1, 1), (3, 1, 1), (3, 3, 1)],
    ["user_id", "item_id", "rating"]
)

# spark dataframe with user/item tuples
test_df = spark.createDataFrame(
    [(1, 1, 1), (3, 3, 1)],
    ["user_id", "item_id", "rating"],
)

# To use C++ based fast prediction, a local cache directory needs to be
# specified.
# * On local machine, `cache_path` can be any valid directories. For example,
#
#   ```python
#   model = SARPlus(
#       spark,
#       col_user="user_id",
#       col_item="item_id",
#       col_rating="rating",
#       col_timestamp="timestamp",
#       similarity_type="jaccard",
#       cache_path="cache",
#   )
#   ```
#
# * On Databricks, `cache_path` needs to be mounted on DBFS.  For example,
#
#   ```python
#   model = SARPlus(
#       spark,
#       col_user="user_id",
#       col_item="item_id",
#       col_rating="rating",
#       col_timestamp="timestamp",
#       similarity_type="jaccard",
#       cache_path="dbfs:/mnt/sarpluscache/cache",
#   )
#   ```
#
# * On Azure Synapse, `cache_path` needs to be mounted on Spark pool's driver
#   node.  For example,
#
#   ```python
#   model = SARPlus(
#       spark,
#       col_user="user_id",
#       col_item="item_id",
#       col_rating="rating",
#       col_timestamp="timestamp",
#       similarity_type="jaccard",
#       cache_path=f"synfs:/{job_id}/mnt/sarpluscache/cache",
#   )
#   ```
#
#   where `job_id` can be obtained by
#
#   ```python
#   from notebookutils import mssparkutils
#   job_id = mssparkutils.env.getJobId()
#   ```
model = SARPlus(
    spark,
    col_user="user_id",
    col_item="item_id",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard",
)
model.fit(train_df)

# To use C++ based fast prediction, the `use_cache` parameter of
# `SARPlus.recommend_k_items()` also needs to be set to `True`.
#
# ```
# model.recommend_k_items(test_df, top_k=3, use_cache=True).show()
# ```
model.recommend_k_items(test_df, top_k=3, remove_seen=False).show()
```

### Jupyter Notebook

Insert this cell prior to the code above.

```python
import os

SARPLUS_MVN_COORDINATE = "com.microsoft.sarplus:sarplus_2.12:0.6.0"
SUBMIT_ARGS = f"--packages {SARPLUS_MVN_COORDINATE} pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("sample")
    .master("local[*]")
    .config("memory", "1G")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.sql.crossJoin.enabled", True)
    .config("spark.sql.sources.default", "parquet")
    .config("spark.sql.legacy.createHiveTableByDefault", True)
    .config("spark.ui.enabled", False)
    .getOrCreate()
)
```


### PySpark Shell

```bash
SARPLUS_MVN_COORDINATE="com.microsoft.sarplus:sarplus_2.12:0.6.0"

# Install pysarplus
pip install pysarplus

# Specify sarplus maven coordinate and configure Spark environment
pyspark --packages "${SARPLUS_MVN_COORDINATE}" \
        --conf spark.sql.crossJoin.enabled=true \
        --conf spark.sql.sources.default=parquet \
        --conf spark.sql.legacy.createHiveTableByDefault=true
```


### Databricks

#### Install libraries

1. Navigate to your Databricks Workspace
1. Create Library
1. Under `Library Source` select `Maven`
1. Enter into `Coordinates`:
   * `com.microsoft.sarplus:sarplus_2.12:0.6.0`
   * or `com.microsoft.sarplus:sarplus-spark-3-2-plus_2.12:0.6.0` (if
     you're on Spark 3.2+)
1. Hit `Create`
1. Attach to your cluster
1. Create 2nd library
1. Under `Library Source` select `PyPI`
1. Enter `pysarplus==0.6.0`
1. Hit `Create`

This will install C++, Python and Scala code on your cluster.  See
[Libraries](https://docs.microsoft.com/en-us/azure/databricks/libraries/)
for details on how to install libraries on Azure Databricks.


#### Configurations

1. Navigate to your Databricks Compute
1. Navigate to your cluster's `Configuration` -> `Advanced options` ->
   `Spark`
1. Put the following configurations into `Spark config`

   ```
   spark.sql.crossJoin.enabled true
   spark.sql.sources.default parquet
   spark.sql.legacy.createHiveTableByDefault true
   ```

These will set the crossJoin property to enable calculation of the
similarity matrix, and set default sources to parquet.


#### Prepare local file system for cache

`pysarplus.SARPlus.recommend_k_items()` needs a local file system path
as its second parameter for storing intermediate files during its
calculation, so you'll also have to **mount** shared storage.

For example, you can [create a storage
account](https://ms.portal.azure.com/#create/Microsoft.StorageAccount)
(e.g. `sarplusstorage`) and a container (e.g. `sarpluscache`) in the
storage account, copy the access key of the storage account, and then
run the following code to mount the storage.

```python
dbutils.fs.mount(
  source = "wasbs://<container>@<storage-account>.blob.core.windows.net",
  mount_point = "/mnt/<container>",
  extra_configs = {
    "fs.azure.account.key.<storage-account>.blob.core.windows.net":
    "<access-key>"
  }
)
```

where `<storage-account>`, `<container>` and `<access-key>` should be
replaced with the actual values, such as `sarplusstorage`,
`sarpluscache` and the access key of the storage account.  Then pass
`"dbfs:/mnt/<container>/cache"` to
`pysarplus.SARPlus.recommend_k_items()` as the value for its 2nd
parameter.


To disable logging messages:

```python
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
```


### Azure Synapse

#### Install libraries

1. Download pysarplus TAR file from
   [pysarplus@PyPI](https://pypi.org/project/pysarplus/)
1. Download sarplus JAR file from 
1. Navigate to your Azure Synapse workspace -> `Manage` -> `Workspace
   packages`
1. Upload pysarplus TAR file and sarplus JAR file as workspace
   packages
1. Navigate to your Azure Synapse workspace -> `Manage` -> `Apache
   Spark pools`
1. Find the Spark pool to install the packages -> `...` -> `Packages`
   -> `Workspace packages` -> `+ Select from workspace packages` and
   select pysarplus TAR file and sarplus JAR file uploaded in the
   previous step
1. Apply

See [Manage libraries for Apache Spark in Azure Synapse
Analytics](https://docs.microsoft.com/en-us/azure/synapse-analytics/spark/apache-spark-azure-portal-add-libraries)
for details on how to manage libraries in Azure Synapse.


#### Prepare local file system for cache

`pysarplus.SARPlus.recommend_k_items()` needs a local file system path
as its second parameter for storing intermediate files during its
calculation, so you'll also have to **mount** shared storage.

For example, you can run the following code to mount the file system
(container) of the default/primary storage account.

```python
from notebookutils import mssparkutils
mssparkutils.fs.mount(
    "abfss://<container>@<storage-account>.dfs.core.windows.net",
    "/mnt/<container>",
    { "linkedService": "<storage-linked-service>"}
)
job_id = mssparkutils.env.getJobId()
```

Then pass `f"synfs:/{job_id}/mnt/<container>/cache"` to
`pysarplus.SARPlus.recommend_k_items()` as the value for its 2nd
parameter.  **NOTE**: `job_id` should be prepended to the local path.

See [How to use file mount/unmount API in Synapse](https://docs.microsoft.com/en-us/azure/synapse-analytics/spark/synapse-file-mount-api)
for more details.


## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for implementation details and
development information.
