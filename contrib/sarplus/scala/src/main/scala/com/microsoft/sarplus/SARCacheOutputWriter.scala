package com.microsoft.sarplus

import java.io.{DataOutputStream, FileInputStream, FileOutputStream, BufferedOutputStream, OutputStream}

import org.apache.hadoop.mapreduce.TaskAttemptContext
import org.apache.spark.sql.execution.datasources.OutputWriter
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import org.apache.commons.io.IOUtils
import com.google.common.io.LittleEndianDataOutputStream

class SARCacheOutputWriter(
    path: String,
    outputStream: OutputStream,
    schema: StructType) extends OutputWriter
{
  // some schema validation
  if (schema.length < 3)
    throw new IllegalArgumentException("Schema must have at least 3 fields")

  val pathOffset = path + ".offsets"
  val pathRelated = path + ".related"

  // temporary output files
  val tempOutputOffset = new LittleEndianDataOutputStream(new BufferedOutputStream(new FileOutputStream(pathOffset), 8*1024))
  val tempOutputRelated = new LittleEndianDataOutputStream(new BufferedOutputStream(new FileOutputStream(pathRelated), 8*1024))

  // On databricks the above 2 files aren't copiedFinal
  val outputFinal = new LittleEndianDataOutputStream(outputStream)

  var lastId = Long.MaxValue

  var rowNumber = 0L
  var offsetCount = 0L

  // this is the new api in spark 2.2+
  def write(row: InternalRow): Unit = {
    // i1, i2, value
    val i1 = row.getLong(0)
    val i2 = row.getLong(1)
    val value = row.getDouble(2)

    if(lastId != i1)
    {
        tempOutputOffset.writeLong(rowNumber) 
        offsetCount += 1
        lastId = i1
    }

    tempOutputRelated.writeInt(i2.toInt)
    tempOutputRelated.writeFloat(value.toFloat)

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
        tempOutputOffset.writeLong(rowNumber) 
        offsetCount += 1
        lastId = i1
    }

    tempOutputRelated.writeInt(i2.toInt)
    tempOutputRelated.writeFloat(value.toFloat)

    rowNumber += 1
  }

  override def close(): Unit = 
  {
      tempOutputOffset.writeLong(rowNumber)
      offsetCount += 1

      tempOutputOffset.close
      tempOutputRelated.close

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
