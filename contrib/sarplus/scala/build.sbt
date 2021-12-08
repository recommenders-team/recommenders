name := "sarplus"


// Denpendency configuration

lazy val sparkVer = settingKey[String]("spark version")
lazy val hadoopVer = settingKey[String]("hadoop version")

lazy val commonSettings = Seq(
  organization := "sarplus.microsoft",
  version := sys.env.getOrElse("VERSION", "0.5.0"),
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases"),
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full),
  sparkVer := sys.env.getOrElse("SPARK_VERSION", "3.2.0"),
  hadoopVer := sys.env.getOrElse("HADOOP_VERSION", "3.3.1"),
  libraryDependencies ++= Seq(
    "com.fasterxml.jackson.core" % "jackson-databind" % "2.12.2",
    "commons-io" % "commons-io" % "2.8.0",
    "org.apache.hadoop" % "hadoop-common" % hadoopVer.value,
    "org.apache.hadoop" % "hadoop-hdfs" % hadoopVer.value,
    "org.apache.spark" %% "spark-core" % sparkVer.value,
    "org.apache.spark" %% "spark-mllib" % sparkVer.value,
    "org.apache.spark" %% "spark-sql" % sparkVer.value,
    "org.scala-lang" % "scala-reflect" % scalaVersion.value,
    "com.google.guava" % "guava" % "15.0",
    "org.scalamock" %% "scalamock" % "4.1.0" % "test",
    "org.scalatest" %% "scalatest" % "3.0.8" % "test",
    "xerces" % "xercesImpl" % "2.12.1",
  ),
  artifactName := {
    (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
      artifact.name + "_" + sv.full + "_s" + sparkVer.value + "_h" + hadoopVer.value + "-" + module.revision + "." + artifact.extension
  },
)

lazy val compat = project.settings(commonSettings)
lazy val root = (project in file("."))
  .dependsOn(compat)
  .settings(
    name := "sarplus",
    commonSettings,
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
    name = "Markus Cozowicz",
    email = "marcozo@microsoft.com"
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


// PGP configuration

credentials += Credentials(
  "GnuPG Key ID",
  "gpg",
  "C72E596B384EC14CFA65D80A36CB250AF1C18ECE",
  "ignored"
)
