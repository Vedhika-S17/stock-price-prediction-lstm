import yfinance as yf
import pandas as pd

def download_stock_data(ticker="AAPL", start_date="2024-03-02", end_date="2025-03-02"):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.to_csv("data/AAPL_RAW.csv")
    print("✅ Data saved to data/AAPL_RAW.csv")

if __name__ == "__main__":
    download_stock_data()
