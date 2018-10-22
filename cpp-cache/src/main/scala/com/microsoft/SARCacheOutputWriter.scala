package com.microsoft

import org.apache.hadoop.mapreduce.{RecordWriter, TaskAttemptContext, TaskAttemptID}
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import java.io.{DataOutputStream, FileOutputStream}

import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.datasources.OutputWriter
import org.apache.spark.sql.types._

class SARCacheOutputWriter(
    path: String,
    context: TaskAttemptContext,
    schema: StructType) extends OutputWriter
{

  val outputOffset = new DataOutputStream(new FileOutputStream(path + java.io.File.separatorChar + "offsets.sar"))

  val outputRelated = new DataOutputStream(new FileOutputStream(path + java.io.File.separatorChar + "related.sar"))

  var lastId = Long.MaxValue

  var rowNumber = 0

  // TODO: validate schema

   // this is the new api in spark 2.2+
  def write(row: InternalRow): Unit = {
        // i1, i2, value
        val i1 = row.getLong(0)
        val i2 = row.getLong(1)
        val value = row.getDouble(2)

        if(lastId != i1)
        {
            outputOffset.writeInt(i1.toInt) 
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
            outputOffset.writeInt(i1.toInt) 
            lastId = i1
        }

        outputRelated.writeInt(i2.toInt)
        outputRelated.writeFloat(value.toFloat)

        rowNumber += 1
  }

  override def close(): Unit = 
  {
      outputOffset.close()
      outputRelated.close()
  } 
}