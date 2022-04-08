/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

import Utils._

// Denpendency configuration

lazy val sparkVer = settingKey[String]("spark version")
lazy val hadoopVer = settingKey[String]("hadoop version")

lazy val commonSettings = Seq(
  version := IO.read(new File("../VERSION")),
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases"),
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full),
  sparkVer := sys.env.getOrElse("SPARK_VERSION", "3.2.1"),
  hadoopVer := sys.env.getOrElse("HADOOP_VERSION", "3.3.1"),
  libraryDependencies ++= Seq(
    "com.fasterxml.jackson.core" % "jackson-databind" % "2.12.3",
    "commons-io" % "commons-io" % "2.11.0",
    "org.apache.hadoop" % "hadoop-common" % hadoopVer.value,
    "org.apache.hadoop" % "hadoop-hdfs" % hadoopVer.value,
    "org.apache.spark" %% "spark-core" % sparkVer.value,
    "org.apache.spark" %% "spark-mllib" % sparkVer.value,
    "org.apache.spark" %% "spark-sql" % sparkVer.value,
    "org.scala-lang" % "scala-reflect" % scalaVersion.value,
    "com.google.guava" % "guava" % "28.0-jre",
    "org.scalamock" %% "scalamock" % "5.2.0" % "test",
    "org.scalatest" %% "scalatest" % "3.0.8" % "test",
    "xerces" % "xercesImpl" % "2.12.2",
  ),
  name := {
    val splitVer = sparkVer.value.split('.')
    val major = splitVer(0).toInt
    val minor = splitVer(1).toInt
    if (major >=3 && minor >= 2) "sarplus-spark-3.2-plus" else "sarplus"
  }
)

lazy val compat = project.settings(commonSettings)
lazy val root = (project in file(".")).dependsOn(compat % "compile-internal").settings(
  commonSettings,
  Compile / packageBin / mappings ++= (compat / Compile / packageBin / mappings).value,
  Compile / packageSrc / mappings ++= (compat / Compile / packageSrc / mappings).value,
)


// POM metadata configuration.  See https://www.scala-sbt.org/release/docs/Using-Sonatype.html

organization := "com.microsoft.sarplus"
organizationName := "microsoft"
organizationHomepage := Some(url("https://microsoft.com"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/microsoft/recommenders/tree/main/contrib/sarplus"),
    "scm:git@github.com:microsoft/recommenders.git"
  )
)

developers := List(
  Developer(
    id = "recodev",
    name = "RecoDev Team at Microsoft",
    email = "recodevteam@service.microsoft.com",
    url = url("https://github.com/microsoft/recommenders/")
  )
)

description := "sarplus"
licenses := Seq("MIT" -> url("http://opensource.org/licenses/MIT"))
homepage := Some(url("https://github.com/microsoft/recommenders/tree/main/contrib/sarplus"))
pomIncludeRepository := { _ => false }
publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
  else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}
publishMavenStyle := true
