from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, avg, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# Define schema (updated for Finnhub format)
schema = StructType([
    StructField("symbol", StringType()),
    StructField("timestamp", LongType()),  # This is epoch milliseconds
    StructField("open", DoubleType()),
    StructField("high", DoubleType()),
    StructField("low", DoubleType()),
    StructField("close", DoubleType()),
    StructField("volume", DoubleType()),
    StructField("previous_close", DoubleType())
])

# Initialize Spark with proper chaining
spark = SparkSession.builder \
    .appName("FinnhubStockProcessor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
    .getOrCreate()

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "stock_prices") \
    .load()

# Process data with timestamp conversion
parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*") \
    .withColumn("event_time", (col("timestamp")/1000).cast("timestamp"))  # Convert ms to timestamp

# Calculate simple moving average with proper time handling
processed_df = parsed_df.withWatermark("event_time", "1 minute") \
    .groupBy(
        window(col("event_time"), "5 minutes"),  # 5-minute windows
        col("symbol")
    ) \
    .agg(avg("close").alias("moving_avg"))

# Output to console and parquet
console_query = processed_df.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .start()

parquet_query = parsed_df.writeStream \
    .format("parquet") \
    .option("path", "/app/data/stock_parquet") \
    .option("checkpointLocation", "/app/data/checkpoint") \
    .start()

# Wait for termination
spark.streams.awaitAnyTermination()