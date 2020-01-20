// You may use this file to add plugin dependencies for sbt.
resolvers += "Spark Package Main Repo" at "https://dl.bintray.com/spark-packages/maven"

addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.6")
