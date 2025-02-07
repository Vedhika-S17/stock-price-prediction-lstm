import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load preprocessed data
X = np.load("data/X.npy")
y_high = np.load("data/y_high.npy")
y_low = np.load("data/y_low.npy")

# Split Data into Training & Validation Sets
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train_high, y_val_high = y_high[:train_size], y_high[train_size:]
y_train_low, y_val_low = y_low[:train_size], y_low[train_size:]

# Define LSTM Model Architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(2)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model
model.fit(X_train, np.column_stack((y_train_high, y_train_low)), epochs=50, batch_size=32,
          validation_data=(X_val, np.column_stack((y_val_high, y_val_low))))

# ✅ FIX: Save model using `.keras` format instead of `.h5`
model.save("models/lstm_model_initial.keras")
print("✅ Initial one-year trained model saved as lstm_model_initial.keras")

# ✅ FIX: Check if TensorFlow is using GPU or CPU
print("TensorFlow running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")
