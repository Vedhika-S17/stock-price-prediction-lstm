import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# === Parameters ===
sequence_length = 60  # 60 days lookback
feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume']

# Load models and scalers
model_high = load_model('models/model_high_price.keras')
model_low = load_model('models/model_low_price.keras')
feature_scaler = joblib.load('data/feature_scaler.pkl')
high_scaler = joblib.load('data/high_scaler.pkl')
low_scaler = joblib.load('data/low_scaler.pkl')

# === Function to Get Data for a Given Date ===
def get_data_for_date(date, df):
    # Find the last row before the given date
    df_before_date = df[df['Date'] < date]
    
    if df_before_date.empty:
        raise ValueError(f"No data found before the date {date}")
    
    # Get the last `sequence_length` days of data (60 days)
    last_row_idx = df_before_date.index[-1]  # last row before the given date
    start_idx = max(0, last_row_idx - sequence_length + 1)  # Ensure we don't go below index 0
    sequence_data = df.iloc[start_idx:last_row_idx + 1][feature_cols].values
    
    # Scale the features using the scaler used during training
    scaled_sequence = feature_scaler.transform(sequence_data)
    
    return scaled_sequence.reshape(1, len(sequence_data), len(feature_cols))

# === Predict High and Low Prices ===
def predict_price(date, df):
    # Prepare the data for the given date
    X = get_data_for_date(date, df)
    
    # Make predictions using the trained models
    pred_high = model_high.predict(X)
    pred_low = model_low.predict(X)
    
    # Inverse transform the predictions to original price scale
    pred_high_inv = high_scaler.inverse_transform(pred_high.reshape(-1, 1)).flatten()[0]
    pred_low_inv = low_scaler.inverse_transform(pred_low.reshape(-1, 1)).flatten()[0]
    
    return pred_high_inv, pred_low_inv

# === Main ===
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/AAPL_stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # Example date to predict for (you can change this)
    date_input = input("Enter a Date (YYYY-MM-DD): ")  # Modify as needed
    
    try:
        high, low = predict_price(date_input, df)
        print(f"Predicted High Price for {date_input}: {high:.2f}")
        print(f"Predicted Low Price for {date_input}: {low:.2f}")
    except Exception as e:
        print(str(e))
