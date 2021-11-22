# Packaging

For [databricks](https://databricks.com/) to properly install a [C++ extension](https://docs.python.org/3/extending/building.html), one must take a detour through [pypi](https://pypi.org/).
Use [twine](https://github.com/pypa/twine) to upload the package to [pypi](https://pypi.org/).

```bash
cd python

python setup.py sdist

twine upload dist/pysarplus-*.tar.gz
```

On [Spark](https://spark.apache.org/) one can install all 3 components (C++, Python, Scala) in one pass by creating a [Spark Package](https://spark-packages.org/). Documentation is rather sparse. Steps to install

1. Package and publish the [pip package](python/setup.py) (see above)
2. Package the [Spark package](scala/build.sbt), which includes the [Scala formatter](scala/src/main/scala/microsoft/sarplus) and references the [pip package](scala/python/requirements.txt) (see below)
3. Upload the zipped Scala package to [Spark Package](https://spark-packages.org/) through a browser. [sbt spPublish](https://github.com/databricks/sbt-spark-package) has a few [issues](https://github.com/databricks/sbt-spark-package/issues/31) so it always fails for me. Don't use spPublishLocal as the packages are not created properly (names don't match up, [issue](https://github.com/databricks/sbt-spark-package/issues/17)) and furthermore fail to install if published to [Spark-Packages.org](https://spark-packages.org/).  

```bash
cd scala
sbt spPublish
```

## Testing

To test the python UDF + C++ backend

```bash
cd python 
python setup.py install && pytest -s tests/
```

To test the Scala formatter

```bash
cd scala
sbt test
```

(use ~test and it will automatically check for changes in source files, but not build.sbt)


## Notes for Spark 3.x  ##

The code now has been modified to support Spark 3.x, and has been
tested under different versions of Databricks Runtime (including 6.4
Extended Support, 7.3 LTS, 9.1 LTS, 10.0 and 10.1) on Azure Databricks
Service.  But now manual packaging is needed:


```bash
export VERSION=0.5.0
cd python
python setup.py bdist_wheel  # => dist/pysarplus-0.5.0-cp38-cp38-linux_x86_64.whl

export SPARK_VERSION=3.2.0
export HADOOP_VERSION=3.3.1
export SCALA_VERSION=2.12.14
cd scala
sbt ++${SCALA_VERSION} package  # => target/scala-2.12/sarplus_2.12.14_s3.2.0_h3.3.1-0.5.0.jar
```

where `VERSION`, `SPARK_VERSION`, `HADOOP_VERSION`, `SCALA_VERSION`
should be customized as needed.  When running on Spark 3.x, extra
configurations are also required:

```
spark.sql.sources.default parquet
spark.sql.legacy.createHiveTableByDefault true
```
