import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os

# Load your preprocessed dataset
df = pd.read_csv("data/AAPL_processed.csv")  # Make sure it has 'Date', 'High', 'Low'
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Select only High and Low
data = df[["High", "Low"]].values

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.save")

# Sequence creation
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(scaled_data, seq_len)

# Split 65:10:25
total_len = len(X)
train_len = int(total_len * 0.65)
val_len = int(total_len * 0.10)

X_train = X[:train_len]
y_train = y[:train_len]

X_val = X[train_len:train_len + val_len]
y_val = y[train_len:train_len + val_len]

X_test = X[train_len + val_len:]
y_test = y[train_len + val_len:]

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 2)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(2))  # Predict High & Low

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint("models/lstm_model.keras", save_best_only=True, monitor="val_loss", mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=10)

# Train model (MAX 50 epochs)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Save loss graph
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_plot.png")
plt.show()
