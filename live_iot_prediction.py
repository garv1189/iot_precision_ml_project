# 6. live_iot_prediction.py

import paho.mqtt.client as mqtt
import json
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/sensor_data"

# Load the trained model and scaler
model = joblib.load('iot_precision_model.joblib')
scaler = joblib.load('scaler.joblib')


def preprocess_data(data):
    df = pd.DataFrame([data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Scale temperature and humidity
    df[['temperature_scaled', 'humidity_scaled']] = scaler.transform(df[['temperature', 'humidity']])

    # For live data, we'll use the current values as rolling statistics
    df['temp_rolling_mean'] = df['temperature_scaled']
    df['temp_rolling_std'] = 0
    df['humid_rolling_mean'] = df['humidity_scaled']
    df['humid_rolling_std'] = 0

    return df[['temperature_scaled', 'humidity_scaled', 'hour', 'day_of_week',
               'temp_rolling_mean', 'temp_rolling_std', 'humid_rolling_mean', 'humid_rolling_std']]


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    try:
        # Decode the message
        payload = json.loads(msg.payload.decode())

        # Preprocess the data
        processed_data = preprocess_data(payload)

        # Make prediction
        prediction = model.predict(processed_data)

        # Check precision
        if prediction[0] == 1:
            print(f"Anomaly detected in data: {payload}")
        else:
            print(f"Data is within expected precision: {payload}")
    except Exception as e:
        print(f"Error processing message: {e}")


# Set up MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Start the loop
client.loop_forever()