import pandas as pd
from kafka import KafkaConsumer
import json
import time
from cleaning import clean_row
from db import save_message_to_db


def consume_data(topic_name='fintech_topic', kafka_url='kafka:29092'):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=kafka_url,
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print('Consumer initialized')
    print('Listening for messages in topic:', topic_name)
    print(consumer, flush=True)

    while True:
        print("Polling...", flush=True)
        message_pack = consumer.poll(timeout_ms=1000)
        if message_pack:
            for tp, messages in message_pack.items():
                for message in messages:
                    print("Message: ")
                    print(message.value, flush=True)
                    if message.value == 'EOF':
                        print('Stopping consumer', flush=True)
                        consumer.close()
                        return True
                    cleaned = clean_row(
                        message.value, 'fintech_data_MET_P1_52_16824')
                    save_message_to_db(cleaned)
        else:
            print("No messages received", flush=True)
