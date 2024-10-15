# 1. generate_iot_data.py

import numpy as np
import pandas as pd


def generate_iot_data(num_samples=1000):
    timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq='5T')
    temperature = np.random.normal(loc=22, scale=5, size=num_samples)
    humidity = np.random.normal(loc=50, scale=10, size=num_samples)

    # Introduce some anomalies
    anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    temperature[anomaly_indices] += np.random.normal(loc=0, scale=10, size=len(anomaly_indices))
    humidity[anomaly_indices] += np.random.normal(loc=0, scale=20, size=len(anomaly_indices))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity
    })
    return df


if __name__ == "__main__":
    iot_data = generate_iot_data()
    iot_data.to_csv('iot_sensor_data.csv', index=False)
    print("IoT sensor data generated and saved to 'iot_sensor_data.csv'")