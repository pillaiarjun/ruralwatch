# ---- src/streaming/consumer.py ----
# Reads messages from the Kafka topic and writes them in batches
# to a Bronze Delta table. Simulates the ingestion side of a
# real-time quarterly CMS update pipeline.

import json
import sys
import os

# Add the project root to the Python path so we can import spark_session
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.spark_session import get_spark
from kafka import KafkaConsumer
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'hospital-financials'
BRONZE_PATH = 'data/bronze/cms_cost_reports_streaming'

# Write to Delta every N messages instead of one row at a time.
# Spark has high startup overhead per write — batching amortizes that cost.
BATCH_SIZE = 100


def create_consumer():
    """Creates and returns a Kafka consumer.

    auto_offset_reset='earliest': when this consumer group starts for the
    first time, read from the very beginning of the topic rather than
    only new messages. This ensures we process the full batch the producer
    already sent.

    group_id: Kafka tracks the read position (offset) per consumer group.
    If the consumer restarts, it resumes from where it left off rather
    than reprocessing everything.

    consumer_timeout_ms: stop blocking and return after 5 seconds of
    receiving no new messages. This is how the for-loop terminates cleanly
    once the producer has finished sending.
    """
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='ruralwatch-bronze-writer',
        consumer_timeout_ms=5000
    )


def write_batch_to_delta(spark, records, path):
    """Converts a list of record dicts to a Spark DataFrame and appends
    it to the Bronze Delta table.

    mode='append': we are adding new quarterly records, not replacing
    the existing Bronze table. Never use overwrite here.
    """
    import re
    pdf = pd.DataFrame(records).astype(str)

    # Delta Lake rejects column names containing spaces or special characters.
    # Replace any invalid character with an underscore.
    # This matches the Bronze layer convention used in notebook 02.
    pdf.columns = [
        re.sub(r'[ ,;{}()\n\t=]+', '_', col).strip('_')
        for col in pdf.columns
    ]

    df = spark.createDataFrame(pdf)
    (
        df.write
        .format('delta')
        .mode('append')
        .save(path)
    )
    print(f'  Wrote {len(records)} records to {path}')


def run_consumer():
    spark = get_spark()
    consumer = create_consumer()

    buffer = []
    total = 0

    print(f'Listening to Kafka topic: {KAFKA_TOPIC}')
    print(f'Will write to Delta in batches of {BATCH_SIZE}')

    for message in consumer:
        buffer.append(message.value)
        total += 1

        # Once the buffer is full, write to Delta and reset
        if len(buffer) >= BATCH_SIZE:
            write_batch_to_delta(spark, buffer, BRONZE_PATH)
            buffer = []

    # Write any remaining records after the consumer times out
    if buffer:
        write_batch_to_delta(spark, buffer, BRONZE_PATH)

    print(f'Consumer finished. Total records processed: {total}')


if __name__ == '__main__':
    run_consumer()