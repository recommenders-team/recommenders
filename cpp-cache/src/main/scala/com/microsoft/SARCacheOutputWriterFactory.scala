package com.microsoft

import org.apache.hadoop.mapreduce.TaskAttemptContext

import org.apache.spark.sql.execution.datasources.{OutputWriter, OutputWriterFactory}
import org.apache.spark.sql.types._


class SARCacheOutputWriterFactory(schema: StructType) extends OutputWriterFactory {

  override def getFileExtension(context: TaskAttemptContext): String = {
    ".sar"
  }

  override def newInstance(
      path: String,
      dataSchema: StructType,
      context: TaskAttemptContext): OutputWriter = {
    new SARCacheOutputWriter(path, context, schema)
  }
}