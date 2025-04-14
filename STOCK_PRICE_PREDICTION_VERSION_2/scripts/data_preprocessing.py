import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/AAPL_RAW.csv")

    # Print the column names to inspect if they match
    print("Columns in raw data:", df.columns)

    # Check the first few rows of the data to understand the structure
    print("First few rows of the raw data:")
    print(df.head())

    # Assuming 'Date' column is present, ensure it's correctly formatted
    # Adjust if necessary (remove extra spaces, incorrect headers, etc.)
    df.columns = df.columns.str.strip()  # Remove any extra spaces in column names
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        print("The 'Date' column was not found. Please check the raw data.")
        return

    # Use only relevant columns (High, Low, Open, Close, Volume)
    data = df[["High", "Low", "Open", "Close", "Volume"]].values

    # Scale the features (X: Open, Close, Volume) and targets (y: High, Low)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scale X (features) and y (targets)
    scaled_data = scaler_X.fit_transform(data[:, 2:])  # Open, Close, Volume (features)
    scaled_target = scaler_y.fit_transform(data[:, :2])  # High, Low (targets)

    # Create sequences
    def create_sequences(data, target, seq_len):
        X, y_high, y_low = [], [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])  # Sequence of features
            y_high.append(target[i, 0])  # High price
            y_low.append(target[i, 1])   # Low price
        return np.array(X), np.array(y_high), np.array(y_low)

    seq_len = 60
    X, y_high, y_low = create_sequences(scaled_data, scaled_target, seq_len)

    # Save preprocessed data and scalers
    joblib.dump(scaler_X, "models/scaler_X.save")
    joblib.dump(scaler_y, "models/scaler_y.save")

    np.save("models/X.npy", X)
    np.save("models/y_high.npy", y_high)
    np.save("models/y_low.npy", y_low)

if __name__ == "__main__":
    preprocess_data()
