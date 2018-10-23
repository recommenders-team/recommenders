name := "sar-cache"

organization := "com.microsoft"

scalaVersion := "2.11.8"

val sparkVersion = "2.3.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "commons-io" % "commons-io" % "2.6"
  "com.google.guava" % "guava" % "25.0-jre"
)