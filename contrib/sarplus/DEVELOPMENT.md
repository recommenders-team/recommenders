# Packaging

Steps to package and publish (also described in
[sarplus.yml](../../.github/workflows/sarplus.yml)):
1. Package and publish the [pip package](python/setup.py).  For
   [databricks](https://databricks.com/) to properly install a [C++
   extension](https://docs.python.org/3/extending/building.html), one
   must take a detour through [pypi](https://pypi.org/).  Use
   [twine](https://github.com/pypa/twine) to upload the package to
   [pypi](https://pypi.org/).

   ```bash
   # build dependencies
   python -m pip install -U build pip twine

   cd python
   cp ../VERSION ./pysarplus/  # copy version file
   python -m build --sdist
   python -m twine upload dist/*
   ```

2. Package the [Scala package](scala/build.sbt), which includes the
   [Scala formatter](scala/src/main/scala/com/microsoft/sarplus) and
   references the pip package.
   
   ```bash
   export SARPLUS_VERSION=$(cat VERSION)
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
   sbt ++${SCALA_VERSION}! packageDoc
   sbt ++${SCALA_VERSION}! packageSrc
   sbt ++${SCALA_VERSION}! makePom

   # sign with GPG
   cd target/scala-${SCALA_VERSION%.*}
   gpg --import <(cat <<< "${GPG_KEY}")
   for file in {*.jar,*.pom}; do gpg -ab "${file}"; done

   # bundle
   jar cvf sarplus-bundle_2.12-${SARPLUS_VERSION}.jar sarplus_*.jar sarplus_*.pom sarplus_*.asc
   jar cvf sarplus-spark-3.2-plus-bundle_2.12-${SARPLUS_VERSION}.jar sarplus-spark*.jar sarplus-spark*.pom sarplus-spark*.asc
   ```

   where `SPARK_VERSION`, `HADOOP_VERSION`, `SCALA_VERSION` should be
   customized as needed.

3. Upload the zipped Scala package bundle to [Nexus Repository
   Manager](https://oss.sonatype.org/) through a browser (See [publish
   manual](https://central.sonatype.org/publish/publish-manual/)).


## Testing

To test the python UDF + C++ backend

```bash
# dependencies
python -m pip install -U build pip twine
python -m pip install -U flake8 pytest pytest-cov scikit-learn

# build
cd python
cp ../VERSION ./pysarplus/  # version file
python -m build --sdist

# test
pytest ./tests
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
additional package called [Sarplus Spark 3.2
Plus](https://search.maven.org/artifact/com.microsoft.sarplus/sarplus-spark-3-2-plus_2.12)
(with Maven coordinate such as
`com.microsoft.sarplus:sarplus-spark-3-2-plus_2.12:0.6.4`) should be
used if running on Spark 3.2 instead of
[Sarplus](https://search.maven.org/artifact/com.microsoft.sarplus/sarplus_2.12)
(with Maven coordinate like
`com.microsoft.sarplus:sarplus_2.12:0.6.4`).

In addition to `spark.sql.crossJoin.enabled true`, extra
configurations are required when running on Spark 3.x:

```
spark.sql.sources.default parquet
spark.sql.legacy.createHiveTableByDefault true
```
