import pandas as pd
import numpy as np
import os

def create_sequences(data, time_steps=60):
    X, y_high, y_low = [], [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])  
        y_high.append(data[i + time_steps, 1])  # High column index
        y_low.append(data[i + time_steps, 2])   # Low column index
    return np.array(X), np.array(y_high), np.array(y_low)

def preprocess(stock_ticker="AAPL", time_steps=60):
    os.makedirs("data", exist_ok=True)
    path = f"data/{stock_ticker}_processed.csv"
    
    if not os.path.exists(path):
        print(f"❌ Processed file not found: {path}")
        return

    df = pd.read_csv(path, index_col=0)
    data = df.values
    
    X, y_high, y_low = create_sequences(data, time_steps)
    
    np.save(f"data/X_{stock_ticker}.npy", X)
    np.save(f"data/y_high_{stock_ticker}.npy", y_high)
    np.save(f"data/y_low_{stock_ticker}.npy", y_low)

    print(f"✅ Preprocessing complete for {stock_ticker}")
    print(f"X shape: {X.shape}, y_high shape: {y_high.shape}, y_low shape: {y_low.shape}")

if __name__ == "__main__":
    preprocess("AAPL")
