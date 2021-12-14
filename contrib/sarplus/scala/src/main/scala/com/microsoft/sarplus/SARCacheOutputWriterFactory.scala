/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

package com.microsoft.sarplus

import org.apache.hadoop.mapreduce.TaskAttemptContext

import org.apache.spark.sql.execution.datasources.{CodecStreams, OutputWriter, OutputWriterFactory}
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.types._
import java.io.File

class SARCacheOutputWriterFactory(schema: StructType) extends OutputWriterFactory {

  override def getFileExtension(context: TaskAttemptContext): String = {
    ".sar"
  }

  override def newInstance(
      path: String,
      dataSchema: StructType,
      context: TaskAttemptContext): OutputWriter = {
    new File(path).getParentFile.mkdirs

    // created here to make the writer testable
    val outputStream = CodecStreams.createOutputStream(context, new Path(path))

    return new SARCacheOutputWriter(path, outputStream, schema)
  }
}
