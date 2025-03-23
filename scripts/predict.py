import yfinance as yf
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ✅ Load the trained model (only from one-year data)
model = load_model("models/lstm_model_test.keras")
print("✅ Loaded model: models/lstm_model_test.keras")

# ✅ Fetch and preprocess stock data
def fetch_stock_data(stock_ticker, target_date):
    df = yf.download(stock_ticker, period="1y", interval="1d")
    
    if target_date not in df.index:
        print(f"❌ No data available for {target_date}.")
        return None, None
    
    df.dropna(inplace=True)

    # ✅ Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    return np.array([df.values]), scaler

# ✅ Predict high & low prices using the entire model
def predict_prices():
    stock_ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    target_date = input("Enter prediction date (YYYY-MM-DD): ").strip()
    
    X_input, scaler = fetch_stock_data(stock_ticker, target_date)
    
    if X_input is None:
        return None

    print(f"✅ Input Shape for Model: {X_input.shape}")  

    predicted_scaled = model.predict(X_input)
    print(f"✅ Scaled Prediction Output: {predicted_scaled}")

    # Reverse scale the prediction
    dummy_array = np.zeros((predicted_scaled.shape[0], 5))
    dummy_array[:, 1:3] = predicted_scaled
    predicted_actual = scaler.inverse_transform(dummy_array)[:, 1:3]

    high_pred, low_pred = predicted_actual[0]
    print(f"📅 Date: {target_date}")
    print(f"📈 Predicted High: {high_pred:.2f}")
    print(f"📉 Predicted Low: {low_pred:.2f}")

# ✅ Run prediction
if __name__ == "__main__":
    predict_prices()



'''
import yfinance as yf
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ✅ Load the latest trained model (use .keras format)
model_path = "models/lstm_model_latest.keras" if os.path.exists("models/lstm_model_latest.keras") else "models/lstm_model_initial.keras"
model = load_model(model_path)
print(f"✅ Loaded model: {model_path}")

# ✅ Fetch stock data up to a specific target date
def fetch_stock_data_until(stock_ticker="AAPL", target_date="2025-02-10", lookback_days=10):
    df = yf.download(stock_ticker, period="1y", interval="1d")  # Get 1 year of data
    df = df[df.index < target_date]  # Keep only dates before the target date
    df.dropna(inplace=True)

    # ✅ Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

    # ✅ Extract last `lookback_days` before the target date
    df_recent = df.iloc[-lookback_days:]
    
    if len(df_recent) < lookback_days:
        print(f"❌ Not enough data before {target_date}. Need {lookback_days} days.")
        return None, None

    return np.array([df_recent.values]), scaler  # Return formatted input & scaler

# ✅ Predict the high & low prices for a specific date
def predict_next_day(stock_ticker="AAPL", target_date="2025-02-10"):
    X_input, scaler = fetch_stock_data_until(stock_ticker, target_date)
    
    if X_input is None:
        return None

    print(f"✅ Input Shape for Model: {X_input.shape}")  # Should be (1, 10, 5)

    predicted_scaled = model.predict(X_input)
    print(f"✅ Scaled Prediction Output: {predicted_scaled}")

    # Reverse scale the prediction
    dummy_array = np.zeros((2, 5))  # Placeholder for inverse transform
    dummy_array[:, 1:3] = predicted_scaled
    predicted_actual = scaler.inverse_transform(dummy_array)[:, 1:3]

    high_pred, low_pred = predicted_actual[0]
    print(f"📅 Date: {target_date}")
    print(f"📈 Predicted High: {high_pred:.2f}")
    print(f"📉 Predicted Low: {low_pred:.2f}")

    return high_pred, low_pred

# ✅ Run prediction for a specific date
if __name__ == "__main__":
    predict_next_day("AAPL", "2025-02-10")
'''
