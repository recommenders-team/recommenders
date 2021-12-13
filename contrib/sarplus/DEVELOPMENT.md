# Packaging

For [databricks](https://databricks.com/) to properly install a [C++
extension](https://docs.python.org/3/extending/building.html), one
must take a detour through [pypi](https://pypi.org/).  Use
[twine](https://github.com/pypa/twine) to upload the package to
[pypi](https://pypi.org/).

```bash
# build dependencies
python -m pip install -U build pip twine

cd python
cp ../VERSION ./  # version file
python -m build --sdist
python -m twine upload dist/*
```

On [Spark](https://spark.apache.org/) one can install all 3 components
(C++, Python, Scala) in one pass by creating a [Spark
Package](https://spark-packages.org/).  Steps to install

1. Package and publish the [pip package](python/setup.py) (see above)
2. Package the [Spark package](scala/build.sbt), which includes the
   [Scala formatter](scala/src/main/scala/microsoft/sarplus) and
   references the pip package (see below)
3. Upload the zipped Scala package bundle to [Nexus Repository
   Manager](https://oss.sonatype.org/) through a browser.

```bash
export SPARK_VERSION="3.1.2"
export HADOOP_VERSION="2.7.4"
export SCALA_VERSION="2.12.10"
GPG_KEY="<gpg-private-key>"

# generate artifacts
cd scala
sbt ++${SCALA_VERSION}! package
sbt ++${SCALA_VERSION}! packageDoc
sbt ++${SCALA_VERSION}! packageSrc
sbt ++${SCALA_VERSION}! makePom

# generate the artifact (sarplus-*-spark32.jar) for Spark 3.2+
export SPARK_VERSION="3.2.0"
export HADOOP_VERSION="3.3.1"
export SCALA_VERSION="2.12.14"
sbt ++${SCALA_VERSION}! package

# sign with GPG
cd target/scala-${SCALA_VERSION%.*}
gpg --import <(cat <<< "${GPG_KEY}")
for file in {*.jar,*.pom}; do gpg -ab "${file}"; done

# bundle
jar cvf sarplus-bundle_2.12-$(cat ../VERSION).jar *.jar *.pom *.asc
```

where `SPARK_VERSION`, `HADOOP_VERSION`, `SCALA_VERSION` should be
customized as needed.


## Testing

To test the python UDF + C++ backend

```bash
# access token for https://recodatasets.blob.core.windows.net/sarunittest/
ACCESS_TOKEN="<test-data-blob-access-token>"

# build dependencies
python -m pip install -U build pip twine

# build
cd python
cp ../VERSION ./  # version file
python -m build --sdist

# test
pytest --token "${ACCESS_TOKEN}" ./tests
```

To test the Scala formatter

```bash
export SPARK_VERSION=3.2.0
export HADOOP_VERSION=3.3.1
export SCALA_VERSION=2.12.14

cd scala
sbt ++${SCALA_VERSION}! test
```


## Notes for Spark 3.x  ##

The code now has been modified to support Spark 3.x, and has been
tested under different versions of Databricks Runtime (including 6.4
Extended Support, 7.3 LTS, 9.1 LTS, 10.0 and 10.1) on Azure Databricks
Service.  However, there is a breaking change of
[org/apache.spark.sql.execution.datasources.OutputWriter](https://github.com/apache/spark/blob/dc0fa1eef74238d745dabfdc86705b59d95b07e1/sql/core/src/main/scala/org/apache/spark/sql/execution/datasources/OutputWriter.scala#L74)
on **Spark 3.2**, which adds an extra function `path()`, so an
additional JAR file with the classifier `spark32` will be needed if
running on Spark 3.2 (See above for packaging).

Also, extra configurations are also required when running on Spark
3.x:

```
spark.sql.sources.default parquet
spark.sql.legacy.createHiveTableByDefault true
```
