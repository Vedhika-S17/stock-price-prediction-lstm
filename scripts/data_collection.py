import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def fetch_stock_data(stock_ticker="AAPL", start_date="2024-02-03", end_date="2025-02-03"):
    os.makedirs("data", exist_ok=True)  # Ensure 'data/' directory exists

    # Fetch stock data
    df = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

    # Check if data is empty before proceeding
    if df.empty or df.shape[0] == 0:
        print(f"❌ No data found for {stock_ticker} between {start_date} and {end_date}. Exiting...")
        return None  # Exit without saving anything

    # Flatten Multi-Index Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # Keep only the first part (Close, High, etc.)

    # Drop missing values
    df.dropna(inplace=True)

    # Save raw data (before scaling)
    df.to_csv(f"data/{stock_ticker}_raw.csv", index=True)

    # Ensure correct column order
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[expected_cols]  # Reorder correctly

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[expected_cols] = scaler.fit_transform(df[expected_cols])

    # Save processed data
    df.to_csv(f"data/{stock_ticker}_processed.csv", index=True)
    print(f"✅ Data saved successfully for {stock_ticker}")
    return df

if __name__ == "__main__":
    df = fetch_stock_data()
    if df is not None:
        print(df.head())
