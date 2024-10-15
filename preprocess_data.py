 #2. preprocess_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
data = pd.read_csv('iot_sensor_data.csv')

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Normalize numerical features
scaler = StandardScaler()
data[['temperature_scaled', 'humidity_scaled']] = scaler.fit_transform(data[['temperature', 'humidity']])

# Save preprocessed data
data.to_csv('preprocessed_iot_data.csv', index=False)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.joblib')

print("Data preprocessed and saved to 'preprocessed_iot_data.csv'")
print("Scaler saved to 'scaler.joblib'")