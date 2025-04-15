import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.preprocessing import MinMaxScaler

# === Parameters ===
sequence_length = 30  # You can change this to 60 for a 60-day lookback
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
scaled_high = high_scaler.fit_transform(df[['High']].values)
scaled_low = low_scaler.fit_transform(df[['Low']].values)
joblib.dump(high_scaler, 'data/high_scaler.pkl')
joblib.dump(low_scaler, 'data/low_scaler.pkl')

# === Create Sequences ===
X = []
y_high = []
y_low = []

for i in range(sequence_length, len(df)):
    X.append(scaled_features[i - sequence_length:i])  # last `sequence_length` days for input
    y_high.append(scaled_high[i])  # High value for the current day
    y_low.append(scaled_low[i])    # Low value for the current day

X = np.array(X)
y_high = np.array(y_high)
y_low = np.array(y_low)

# === Train-Test Split ===
X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low = train_test_split(
    X, y_high, y_low, test_size=0.2, shuffle=False
)

# === Define Model Factory ===
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=False, input_shape=input_shape))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# === Early Stopping ===
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === Train High Price Model ===
model_high = create_model((X_train.shape[1], X_train.shape[2]))
history_high = model_high.fit(
    X_train, y_train_high,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test_high),
    callbacks=[early_stopping]
)
os.makedirs('models', exist_ok=True)
model_high.save('models/model_high_price.keras')

# === Train Low Price Model ===
model_low = create_model((X_train.shape[1], X_train.shape[2]))
history_low = model_low.fit(
    X_train, y_train_low,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test_low),
    callbacks=[early_stopping]
)
model_low.save('models/model_low_price.keras')

# === Plot Loss Function ===
def plot_loss(history, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{filename}')
    plt.close()

plot_loss(history_high, 'High Price Prediction Loss', 'loss_high.png')
plot_loss(history_low, 'Low Price Prediction Loss', 'loss_low.png')

# === Evaluation ===
test_loss_high = model_high.evaluate(X_test, y_test_high)
test_loss_low = model_low.evaluate(X_test, y_test_low)

print(f"Test Loss (High Price): {test_loss_high:.4f}")
print(f"Test Loss (Low Price): {test_loss_low:.4f}")
