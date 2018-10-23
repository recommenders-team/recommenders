:require /Users/eisber/Documents/Recommenders/cpp-cache/target/scala-2.11/sar-cache_2.11-0.1-SNAPSHOT.jar

var df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/Users/eisber/Documents/Recommenders/cpp-cache/sample.txt")

df.show()
df.schema

df.coalesce(1).write.format("com.microsoft.sar").mode("overwrite").save("/Users/eisber/Documents/Recommenders/cpp-cache/output5")
