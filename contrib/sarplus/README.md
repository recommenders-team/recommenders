# SARplus
pronounced sUrplus as it's simply better if not best!

[![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/contrib%20sarplus?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=107&branchName=master)
[![PyPI version](https://badge.fury.io/py/pysarplus.svg)](https://badge.fury.io/py/pysarplus)

Features
* Scalable PySpark based [implementation](python/pysarplus/SARPlus.py)
* Fast C++ based [predictions](python/src/pysarplus.cpp)
* Reduced memory consumption: similarity matrix cached in-memory once per worker, shared accross python executors 

# Benchmarks

| # Users | # Items | # Ratings | Runtime | Environment | Dataset | 
|---------|---------|-----------|---------|-------------|---------|
| 2.5mio  | 35k     | 100mio    | 1.3h    | Databricks, 8 workers, [Azure Standard DS3 v2](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/) (4 core machines) | |

# Top-K Recommendation Optimization

There are a couple of key optimizations:

* map item ids (e.g. strings) to a continuous set of indexes to optmize storage and simplify access
* convert similarity matrix to exactly the representation the C++ component needs, thus enabling simple shared, memory mapping of the cache file and avoid parsing. This requires a customer formatter, written in Scala
* shared read-only memory mapping allows us to re-use the same memory from multiple python executors on the same worker node
* partition the input test users and past seen items by users, allowing for scale out
* perform as much of the work as possible in PySpark (way simpler)
* top-k computation
** reverse the join by summing reverse joining the users past seen items with any related items
** make sure to always just keep top-k items in-memory
** use standard join using binary search between users past seen items and the related items

![Image of sarplus top-k recommendation optimization](https://recodatasets.blob.core.windows.net/images/sarplus_udf.svg) 

# Usage

```python
from pysarplus import SARPlus

# spark dataframe with user/item/rating/optional timestamp tuples
train_df = spark.createDataFrame([
    (1, 1, 1), 
    (1, 2, 1), 
    (2, 1, 1), 
    (3, 1, 1), 
    (3, 3, 1)], 
    ['user_id', 'item_id', 'rating'])

# spark dataframe with user/item tuples
test_df = spark.createDataFrame([
    (1, 1, 1), 
    (3, 3, 1)], 
    ['user_id', 'item_id', 'rating'])

model = SARPlus(spark, col_user='user_id', col_item='item_id', col_rating='rating', col_timestamp='timestamp', similarity_type='jaccard')
model.fit(train_df)


model.recommend_k_items(test_df, 'sarplus_cache', top_k=3).show()

# For databricks
# model.recommend_k_items(test_df, 'dbfs:/mnt/sarpluscache', top_k=3).show()
```

## Jupyter Notebook

Insert this cell prior to the code above.

```python
import os

SUBMIT_ARGS = "--packages microsoft:sarplus:0.2.6 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("sample")
    .master("local[*]")
    .config("memory", "1G")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.sql.crossJoin.enabled", True)
    .config("spark.ui.enabled", False)
    .getOrCreate()
)
```

## PySpark Shell

```bash
pip install pysarplus
pyspark --packages microsoft:sarplus:0.2.6 --conf spark.sql.crossJoin.enabled=true
```

## Databricks

One must set the crossJoin property to enable calculation of the similarity matrix (Clusters / &lt; Cluster &gt; / Configuration / Spark Config)

```
spark.sql.crossJoin.enabled true
```

1. Navigate to your workspace 
2. Create library
3. Under 'Source' select 'Maven Coordinate'
4. Enter 'microsoft:sarplus:0.2.5' or 'microsoft:sarplus:0.2.6' if you're on Spark 2.4.1
5. Hit 'Create Library'
6. Attach to your cluster
7. Create 2nd library
8. Under 'Source' select 'Upload Python Egg or PyPI'
9. Enter 'pysarplus'
10. Hit 'Create Library'

This will install C++, Python and Scala code on your cluster.

You'll also have to mount shared storage

1. Create [Azure Storage Blob](https://ms.portal.azure.com/#create/Microsoft.StorageAccount-ARM)
2. Create storage account (e.g. <yourcontainer>)
3. Create container (e.g. sarpluscache)

1. Navigate to User / User Settings
2. Generate new token: enter 'sarplus'
3. Use databricks shell (installation here)
4. databricks configure --token
4.1. Host: e.g. https://westus.azuredatabricks.net
5. databricks secrets create-scope --scope all --initial-manage-principal users
6. databricks secrets put --scope all --key sarpluscache
6.1. enter Azure Storage Blob key of Azure Storage created before
7. Run mount code


```pyspark
dbutils.fs.mount(
  source = "wasbs://sarpluscache@<accountname>.blob.core.windows.net",
  mount_point = "/mnt/sarpluscache",
  extra_configs = {"fs.azure.account.key.<accountname>.blob.core.windows.net":dbutils.secrets.get(scope = "all", key = "sarpluscache")})
```

Disable annoying logging

```pyspark
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
```

# Development

See [DEVELOPMENT.md](DEVELOPMENT.md)