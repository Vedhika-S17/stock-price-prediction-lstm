
# data_collection.py

import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker='AAPL', start_date='2024-02-03', end_date='2025-02-05', folder='data'):
    # Download historical stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index to include 'Date' as a column
    data.reset_index(inplace=True)

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Save the data to CSV inside the specified folder
    file_path = os.path.join(folder, f"{ticker}_stock_data.csv")
    data.to_csv(file_path, index=False)
    
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    download_stock_data()
