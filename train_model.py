# 4. train_model.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

# Load data
data = pd.read_csv('featured_iot_data.csv')

# Prepare features and target
features = ['temperature_scaled', 'humidity_scaled', 'hour', 'day_of_week',
            'temp_rolling_mean', 'temp_rolling_std', 'humid_rolling_mean', 'humid_rolling_std']
X = data[features]
y = data['anomaly']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'iot_precision_model.joblib')

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Model trained and saved to 'iot_precision_model.joblib'")