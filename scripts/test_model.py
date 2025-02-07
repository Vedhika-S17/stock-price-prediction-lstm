import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ✅ Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ✅ Load preprocessed data
X = np.load("data/X.npy")
y_high = np.load("data/y_high.npy")
y_low = np.load("data/y_low.npy")

# ✅ Split Data into 75% Training & 25% Testing
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train_high, y_test_high = y_high[:train_size], y_high[train_size:]
y_train_low, y_test_low = y_low[:train_size], y_low[train_size:]

# ✅ Define a New LSTM Model for Testing
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # Input Layer
    LSTM(50, return_sequences=True),  # LSTM Layer 1
    Dropout(0.2),  # Dropout Layer 1
    LSTM(50, return_sequences=False),  # LSTM Layer 2
    Dropout(0.2),  # Dropout Layer 2
    Dense(25, activation='relu'),  # Dense Layer
    Dense(2)  # Output Layer (Predict High & Low Prices)
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='mse')

# ✅ Train Model on 75% Data
model.fit(X_train, np.column_stack((y_train_high, y_train_low)), epochs=50, batch_size=32)

# ✅ Test Model on 25% Data
y_pred = model.predict(X_test)

# ✅ Fix: Manually Compute RMSE Instead of Using `squared=False`
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ✅ Evaluate Model Performance
rmse_high = rmse(y_test_high, y_pred[:, 0])
rmse_low = rmse(y_test_low, y_pred[:, 1])
mae_high = mean_absolute_error(y_test_high, y_pred[:, 0])
mae_low = mean_absolute_error(y_test_low, y_pred[:, 1])

print(f"📊 Model Evaluation:")
print(f"🔹 RMSE High: {rmse_high:.2f}, RMSE Low: {rmse_low:.2f}")
print(f"🔹 MAE High: {mae_high:.2f}, MAE Low: {mae_low:.2f}")

# ✅ Save the New Trained Model
model.save("models/lstm_model_test.keras")
print("✅ Model trained & saved as lstm_model_test.keras")
