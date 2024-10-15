# 5. iot_device_simulator.py

import paho.mqtt.client as mqtt
import json
import time
import random

# MQTT broker settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/sensor_data"

# Connect to MQTT broker
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)


def generate_iot_data():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    temperature = random.uniform(15, 30)  # Temperature between 15°C and 30°C
    humidity = random.uniform(30, 70)  # Humidity between 30% and 70%

    # Occasionally introduce anomalies
    if random.random() < 0.05:  # 5% chance of anomaly
        temperature += random.uniform(-10, 10)
        humidity += random.uniform(-20, 20)

    return {
        "timestamp": timestamp,
        "temperature": temperature,
        "humidity": humidity
    }


def publish_iot_data():
    while True:
        data = generate_iot_data()
        message = json.dumps(data)
        client.publish(MQTT_TOPIC, message)
        print(f"Published: {message}")
        time.sleep(5)  # Wait for 5 seconds before next publication


if __name__ == "__main__":
    try:
        publish_iot_data()
    except KeyboardInterrupt:
        print("Simulation stopped")
    finally:
        client.disconnect()