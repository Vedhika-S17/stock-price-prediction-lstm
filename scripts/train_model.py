import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# Load data
df = pd.read_csv("data/AAPL_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
data = df[["High", "Low"]].values

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, "models/scaler.save")

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(scaled_data, seq_len)

# Split data
train_len = int(len(X) * 0.65)
val_len = int(len(X) * 0.10)

X_train, y_train = X[:train_len], y[:train_len]
X_val, y_val = X[train_len:train_len + val_len], y[train_len:train_len + val_len]
X_test, y_test = X[train_len + val_len:], y[train_len + val_len:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25),
    Dense(2)  # Predicting High and Low
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
