docker-compose up -d

######
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python kafka/producer.py


##### (another tab)
docker exec -it stock-spark-1 spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 /app/processing.py
