import os
from confluent_kafka import Consumer
from dotenv import load_dotenv
import json

load_dotenv()

# Configuration
KAFKA_BROKER = os.getenv('KAFKA_BROKER')
TOPIC_NAME = os.getenv('TOPIC_NAME')
GROUP_ID = 'stock-consumer-group'

# Set up Kafka consumer
conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': GROUP_ID,
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
consumer.subscribe([TOPIC_NAME])

def process_message(msg):
    try:
        data = json.loads(msg.value().decode('utf-8'))
        print(f"Received stock data: {data}")
        # Here you could write to a database or file for Tableau
        with open('data/stock_data.json', 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception as e:
        print(f"Error processing message: {e}")

if __name__ == "__main__":
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            process_message(msg)
    except KeyboardInterrupt:
        print("Consumer stopped")
    finally:
        consumer.close()