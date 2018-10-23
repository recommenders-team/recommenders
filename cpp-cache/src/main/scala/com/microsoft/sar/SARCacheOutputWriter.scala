package com.microsoft.sar

import java.io.{DataOutputStream, FileInputStream, FileOutputStream}

import org.apache.hadoop.mapreduce.{RecordWriter, TaskAttemptContext, TaskAttemptID}
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.datasources.{CodecStreams, OutputWriter}
import org.apache.spark.sql.types._

import org.apache.commons.io.IOUtils
import com.google.common.io.LittleEndianDataOutputStream

class SARCacheOutputWriter(
    path: String,
    context: TaskAttemptContext,
    schema: StructType) extends OutputWriter
{
  val pathOffset = path + ".offsets"
  val pathRelated = path + ".related"

  val outputOffset = new LittleEndianDataOutputStream(new FileOutputStream(pathOffset))
  val outputRelated = new LittleEndianDataOutputStream(new FileOutputStream(pathRelated))

  // On databricks the above 2 files aren't copied
  val outputFinal = new LittleEndianDataOutputStream(CodecStreams.createOutputStream(context, new Path(path)))

  var lastId = Long.MaxValue

  var rowNumber = 0L
  var offsetCount = 0L

  // TODO: validate schema

  // this is the new api in spark 2.2+
  def write(row: InternalRow): Unit = {
    // i1, i2, value
    val i1 = row.getLong(0)
    val i2 = row.getLong(1)
    val value = row.getDouble(2)

    // System.out.println("i1:  "+ i1 + " i2: " + i2 +  " Value:  " + value) // debug
    if(lastId != i1)
    {
        // System.out.println("RowNumber: " + rowNumber) // debug
        outputOffset.writeLong(rowNumber) 
        offsetCount += 1
        lastId = i1
    }

    outputRelated.writeInt(i2.toInt)
    outputRelated.writeFloat(value.toFloat)

    rowNumber += 1
  }

  // api in spark 2.0 - 2.1
  def write(row: Row): Unit = {
    // i1, i2, value
    val i1 = row.getLong(0)
    val i2 = row.getLong(1)
    val value = row.getDouble(2)

    if(lastId != i1)
    {
        outputOffset.writeLong(rowNumber) 
        offsetCount += 1
        lastId = i1
    }

    outputRelated.writeInt(i2.toInt)
    outputRelated.writeFloat(value.toFloat)

    rowNumber += 1
  }

  override def close(): Unit = 
  {
      outputOffset.close
      outputRelated.close

      outputFinal.writeLong(offsetCount)

      var input = new FileInputStream(pathOffset)
      IOUtils.copy(input, outputFinal)
      input.close

      input = new FileInputStream(pathRelated)
      IOUtils.copy(input, outputFinal)
      input.close

      outputFinal.close
  } 
}