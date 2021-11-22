name := "sarplus"
licenses := Seq("MIT" -> url("http://opensource.org/licenses/MIT"))
// credentials += Credentials(Path.userHome / ".m2" / ".sbtcredentials")
// publishTo := {
//   val org = sys.env.getOrElse("ORG", "")
//   val project = sys.env.getOrElse("PROJECT", "")
//   val feed = sys.env.getOrElse("FEED", "")
//   Some("releases" at "https://pkgs.dev.azure.com/%s/%s/_packaging/%s/Maven/v1".format(org, project, feed))
// }

lazy val sparkVer = settingKey[String]("spark version")
lazy val hadoopVer = settingKey[String]("hadoop version")
lazy val commonsIoVer = settingKey[String]("commons-io version")
lazy val jacksonDatabindVer = settingKey[String]("jackson-databind version")

lazy val commonSettings = Seq(
  organization := "sarplus.microsoft",
  version := sys.env.getOrElse("VERSION", "0.5.0"),
  crossScalaVersions := Seq("2.11.12", "2.12.10", "2.12.14"),
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases"),
  ),
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full),
  sparkVer := {
    scalaVersion.value match {
      case "2.11.12" => "2.4.5"
      case "2.12.10" => "3.1.2"
      case "2.12.14" => "3.2.0"
      case _         => "3.2.0"
    }
  },

  hadoopVer := {
    scalaVersion.value match {
      case "2.11.12" => "2.7.3"
      case "2.12.10" => "2.7.4"
      case "2.12.14" => "3.3.1"
      case _         => "3.3.1"
    }
  },

  commonsIoVer := {
    scalaVersion.value match {
      case "2.11.12" => "2.4"
      case "2.12.10" => "2.4"
      case "2.12.14" => "2.8.0"
      case _         => "2.8.0"
    }
  },

  jacksonDatabindVer := {
    scalaVersion.value match {
      case "2.11.12" => "2.6.7.1"
      case "2.12.10" => "2.10.0"
      case "2.12.14" => "2.12.2"
      case _         => "2.12.2"
    }
  },
  libraryDependencies ++= Seq(
    "com.fasterxml.jackson.core" % "jackson-databind" % jacksonDatabindVer.value,
    "commons-io" % "commons-io" % commonsIoVer.value,
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
      artifact.name + "_" + sv.full + "_" + sparkVer.value + "-" + module.revision + "." + artifact.extension
  },
)

lazy val compat = project.settings(commonSettings)
lazy val root = (project in file("."))
  .dependsOn(compat)
  .settings(
    name := "sarplus",
    commonSettings,
  )

// aetherPublishBothSettings
