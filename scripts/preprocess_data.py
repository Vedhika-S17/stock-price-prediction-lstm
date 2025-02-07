import pandas as pd
import numpy as np
import os

def create_sequences(data, time_steps=60):
    X, y_high, y_low = [], [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])  
        y_high.append(data[i + time_steps, 1])  # High price
        y_low.append(data[i + time_steps, 2])   # Low price
    return np.array(X), np.array(y_high), np.array(y_low)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Load processed stock data
    df = pd.read_csv("data/AAPL_processed.csv", index_col=0)

    # Convert dataframe to numpy array
    data = df.values

    # Convert data into time-series format
    time_steps = 60  # Use last 60 days to predict next day
    X, y_high, y_low = create_sequences(data, time_steps)

    # Save preprocessed data as NumPy arrays
    np.save("data/X.npy", X)
    np.save("data/y_high.npy", y_high)
    np.save("data/y_low.npy", y_low)

    print("✅ Data preprocessing complete. Shapes:")
    print(f"X: {X.shape}, y_high: {y_high.shape}, y_low: {y_low.shape}")
