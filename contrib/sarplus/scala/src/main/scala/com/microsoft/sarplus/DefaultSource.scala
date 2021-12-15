/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

package com.microsoft.sarplus

import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.SparkSession
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.fs.FileStatus

import org.apache.spark.sql.execution.datasources.{FileFormat, OutputWriterFactory}

import org.apache.spark.sql.types.StructType

class DefaultSource extends FileFormat with DataSourceRegister {

  override def equals(other: Any): Boolean = other match {
    case _: DefaultSource => true
    case _ => false
  }

  override def shortName(): String = "sar"

  override def inferSchema(
      spark: SparkSession,
      options: Map[String, String],
      files: Seq[FileStatus]): Option[StructType] = {
        return None // TODO: ?
    }

  override def prepareWrite(
      spark: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType): OutputWriterFactory = {

      return new SARCacheOutputWriterFactory(dataSchema)
    }
}
