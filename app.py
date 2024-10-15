# app.py

from flask import Flask, render_template, jsonify
import paho.mqtt.client as mqtt
import json
import pandas as pd
import joblib
from collections import deque
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('iot_precision_model.joblib')
scaler = joblib.load('scaler.joblib')

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/sensor_data"

# Store recent data
data_queue = deque(maxlen=100)


def preprocess_data(data):
    df = pd.DataFrame([data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    df[['temperature_scaled', 'humidity_scaled']] = scaler.transform(df[['temperature', 'humidity']])

    df['temp_rolling_mean'] = df['temperature_scaled']
    df['temp_rolling_std'] = 0
    df['humid_rolling_mean'] = df['humidity_scaled']
    df['humid_rolling_std'] = 0

    return df[['temperature_scaled', 'humidity_scaled', 'hour', 'day_of_week',
               'temp_rolling_mean', 'temp_rolling_std', 'humid_rolling_mean', 'humid_rolling_std']]


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        processed_data = preprocess_data(payload)
        prediction = model.predict(processed_data)

        payload['anomaly'] = int(prediction[0])
        data_queue.append(payload)
    except Exception as e:
        print(f"Error processing message: {e}")


# Set up MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.subscribe(MQTT_TOPIC)
mqtt_client.loop_start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def get_data():
    return jsonify(list(data_queue))


@app.route('/plot-data')
def plot_data():
    df = pd.DataFrame(list(data_queue))

    if df.empty:
        return jsonify({})

    temp_trace = go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines+markers', name='Temperature')
    humid_trace = go.Scatter(x=df['timestamp'], y=df['humidity'], mode='lines+markers', name='Humidity')

    anomaly_df = df[df['anomaly'] == 1]
    anomaly_trace = go.Scatter(x=anomaly_df['timestamp'], y=anomaly_df['temperature'],
                               mode='markers', name='Anomaly',
                               marker=dict(color='red', size=10, symbol='star'))

    layout = go.Layout(title='IoT Sensor Data', xaxis=dict(title='Timestamp'), yaxis=dict(title='Value'))
    fig = go.Figure(data=[temp_trace, humid_trace, anomaly_trace], layout=layout)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graphJSON)


if __name__ == '__main__':
    app.run(debug=True)