# ---- src/streaming/producer.py ----
# Reads a quarterly CMS cost report batch and sends each row to Kafka
# as a JSON message. Simulates a quarterly CMS data release arriving
# in real time.

import json
import time
import pandas as pd
from kafka import KafkaProducer

# ── Configuration ──────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'hospital-financials'

# We use the 2022 CMS file as our simulated "new quarterly release"
BATCH_FILE = 'data/raw/cms/CostReport_2022_Final.csv'


def create_producer():
    """Creates and returns a Kafka producer.

    value_serializer: converts each Python dict to UTF-8 encoded JSON bytes
    before sending. Kafka transmits raw bytes, not Python objects.

    key_serializer: converts the message key (hospital CCN string) to bytes.
    Kafka routes messages with the same key to the same partition, so all
    filings for a given hospital always land in the same partition in order.

    bootstrap_servers: address of the Kafka broker running in Docker.
    """
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8')
    )


def send_quarterly_batch(producer, filepath, delay_seconds=0.01):
    """Reads a quarterly CMS CSV and sends each row as a Kafka message.

    delay_seconds: pause between messages to simulate records arriving
    over time rather than all at once.
    """
    df = pd.read_csv(filepath, low_memory=False)
    print(f'Loaded {len(df)} hospital records from {filepath}')
    print(f'Sending to Kafka topic: {KAFKA_TOPIC}')

    sent = 0
    for _, row in df.iterrows():
        # Convert row to dict; replace NaN with None so json.dumps works
        # (JSON has null; Python's NaN is not valid JSON)
        record = row.where(row.notna(), other=None).to_dict()

        # Use Provider CCN as the message key
        key = str(record.get('Provider CCN', 'unknown'))

        # .send() is non-blocking — it buffers the message internally
        producer.send(KAFKA_TOPIC, key=key, value=record)
        sent += 1

        if sent % 500 == 0:
            print(f'  Sent {sent} records...')

        time.sleep(delay_seconds)

    # .flush() blocks until all buffered messages are actually delivered
    producer.flush()
    print(f'Done. {sent} records sent to topic: {KAFKA_TOPIC}')


if __name__ == '__main__':
    producer = create_producer()
    send_quarterly_batch(producer, BATCH_FILE)