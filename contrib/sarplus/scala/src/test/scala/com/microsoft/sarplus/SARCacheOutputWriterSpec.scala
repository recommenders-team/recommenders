package com.microsoft.sarplus

import org.scalatest._
import java.io.{File, FileInputStream, FileOutputStream}
import java.nio.file.Files
import org.scalamock.scalatest.MockFactory
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import com.google.common.io.LittleEndianDataInputStream

trait WriterBehavior { this: FlatSpec with MockFactory => 

  def createRow(i1: Long, i2: Long, value: Double): InternalRow = {
    val row = stub[InternalRow]

    (row.getLong _).when(0).returns(i1)
    (row.getLong _).when(1).returns(i2)
    (row.getDouble _).when(2).returns(value)

    row
  }

  def createInternalRow(i1: Long, i2: Long, value: Double): Row = {
    val row = stub[Row]

    (row.getLong _).when(0).returns(i1)
    (row.getLong _).when(1).returns(i2)
    (row.getDouble _).when(2).returns(value)

    row
  }

  def writeCache(internalRow: Boolean) {
    it should "successfully write and read the cache " in {
      val tempDir = Files.createTempDirectory("sarplus")
      val outputFile = File.createTempFile("sarplus", "sar")
      val outputStream = new FileOutputStream(outputFile)

      val schema = StructType(
          StructField("i1", LongType, true) :: 
          StructField("i2", LongType, true) :: 
          StructField("value", DoubleType, true) :: 
          Nil)

      val writer = new SARCacheOutputWriter(tempDir.toString, outputStream, schema)

      // poor mans templates
      if (internalRow) {
        writer.write(createInternalRow(0,0,0.2))
        writer.write(createInternalRow(0,1,0.1))
        writer.write(createInternalRow(1,1,0.3))
      }
      else {
        writer.write(createRow(0,0,0.2))
        writer.write(createRow(0,1,0.1))
        writer.write(createRow(1,1,0.3))
      }

      writer.close

      // validation
      val reader = new LittleEndianDataInputStream(new FileInputStream(outputFile))
      
      // 2 item ids + 1 for to final element (support offset[i] and offset[i+1] without checks)
      assert(3 === reader.readLong)

      // 3 offsets
      assert(0 === reader.readLong)
      assert(2 === reader.readLong)
      assert(3 === reader.readLong)

      // 3 values
      assert(0 === reader.readInt)
      assert(0.2f === reader.readFloat)
      assert(1 === reader.readInt)
      assert(0.1f === reader.readFloat)
      assert(1 === reader.readInt)
      assert(0.3f === reader.readFloat)
    }
  }
}

class SARCacheOutputWriterSpec extends FlatSpec with MockFactory with WriterBehavior {
  "SARCacheOutputWriter for Row" should behave like writeCache(false)
  "SARCacheOutputWriter for InternalRow" should behave like writeCache(true)
}