import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# === Parameters ===
sequence_length = 30
feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
data_path = 'data/AAPL_stock_data.csv'

# === Load Data ===
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# === Feature Scaling ===
feature_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(df[feature_cols])
joblib.dump(feature_scaler, 'data/feature_scaler.pkl')

# === Target Scaling ===
high_scaler = MinMaxScaler()
low_scaler = MinMaxScaler()
scaled_high = high_scaler.fit_transform(df[['High']][sequence_length:])
scaled_low = low_scaler.fit_transform(df[['Low']][sequence_length:])
joblib.dump(high_scaler, 'data/high_scaler.pkl')
joblib.dump(low_scaler, 'data/low_scaler.pkl')

# === Create Sequences ===
X = []
y_high = []
y_low = []

for i in range(sequence_length, len(df) - 1):
    X.append(scaled_features[i - sequence_length:i])
    y_high.append(df['High'].iloc[i + 1])  # predict next day's high
    y_low.append(df['Low'].iloc[i + 1])    # predict next day's low

X = np.array(X)
y_high = high_scaler.transform(np.array(y_high).reshape(-1, 1)).flatten()
y_low = low_scaler.transform(np.array(y_low).reshape(-1, 1)).flatten()

# === Save Preprocessed Data ===
os.makedirs('data', exist_ok=True)
np.save('data/scaled_data.npy', X)
np.save('data/y_high.npy', y_high)
np.save('data/y_low.npy', y_low)

print("✅ Preprocessing complete.")
print(f"📦 Scaled data shape: {X.shape}")
print(f"🎯 Target output shape (y_high): {y_high.shape}")
print(f"🎯 Target output shape (y_low): {y_low.shape}")
