# 3. feature_engineering.py

import pandas as pd
import numpy as np

# Load preprocessed data
data = pd.read_csv('preprocessed_iot_data.csv')

# Calculate rolling statistics
window_size = 12  # 1-hour window (assuming 5-minute intervals)
data['temp_rolling_mean'] = data['temperature_scaled'].rolling(window=window_size).mean()
data['temp_rolling_std'] = data['temperature_scaled'].rolling(window=window_size).std()
data['humid_rolling_mean'] = data['humidity_scaled'].rolling(window=window_size).mean()
data['humid_rolling_std'] = data['humidity_scaled'].rolling(window=window_size).std()

# Fill NaN values created by rolling calculations
data = data.fillna(method='bfill')

# Create target variable (anomaly)
data['anomaly'] = np.where(
    (np.abs(data['temperature_scaled']) > 3) | (np.abs(data['humidity_scaled']) > 3),
    1, 0
)

# Save feature-engineered data
data.to_csv('featured_iot_data.csv', index=False)
print("Feature engineering completed. Data saved to 'featured_iot_data.csv'")