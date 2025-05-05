import os
import requests
import time
import json
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

# Configuration
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
KAFKA_BROKER = os.getenv('KAFKA_BROKER')
TOPIC_NAME = os.getenv('TOPIC_NAME')

# Finnhub API endpoint
FINNHUB_URL = "https://finnhub.io/api/v1/quote"

# Kafka Producer Setup
conf = {'bootstrap.servers': KAFKA_BROKER}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def fetch_stock_data(symbol):
    try:
        params = {
            'symbol': symbol,
            'token': FINNHUB_API_KEY
        }
        response = requests.get(FINNHUB_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Transform to match Alpha Vantage format
        message = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),  # Finnhub uses UNIX ms
            'open': data.get('o', 0),
            'high': data.get('h', 0),
            'low': data.get('l', 0),
            'close': data.get('c', 0),
            'volume': data.get('v', 0),
            'previous_close': data.get('pc', 0)  # Additional Finnhub field
        }
        return message
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Example symbols
    
    while True:
        for symbol in symbols:
            data = fetch_stock_data(symbol)
            if data:
                producer.produce(
                    TOPIC_NAME,
                    json.dumps(data).encode('utf-8'),
                    callback=delivery_report
                )
                producer.flush()
        time.sleep(60)  # Finnhub free tier: 60 calls/minute