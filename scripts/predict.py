import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ✅ Load model
model = load_model("models/lstm_model_test.keras")
print("✅ Model loaded.")

# ✅ Fetch stock data up to the prediction date
def fetch_data_up_to_date(target_date):
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD.")
        return None, None

    # Fetch data up to target date
    end_date = target.strftime("%Y-%m-%d")
    df = yf.download("AAPL", start=(target - timedelta(days=120)).strftime("%Y-%m-%d"), end=end_date, interval="1d")
    df.dropna(inplace=True)

    if len(df) < 60:
        print(f"❌ Not enough data to make prediction. Only {len(df)} days available.")
        return None, None

    df_recent = df[-60:]  # last 60 days
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df_recent.copy()
    df_scaled[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
        df_scaled[['Open', 'High', 'Low', 'Close', 'Volume']]
    )

    return np.array([df_scaled.values]), scaler

# ✅ Predict high & low
def predict_prices():
    target_date = input("Enter prediction date (YYYY-MM-DD): ").strip()
    X, scaler = fetch_data_up_to_date(target_date)
    if X is None:
        return

    pred_scaled = model.predict(X)

    dummy = np.zeros((1, 5))
    dummy[:, 1:3] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[:, 1:3]

    high, low = pred_actual[0]
    print(f"\n📅 Prediction for: {target_date}")
    print(f"📈 Predicted High: {high:.2f}")
    print(f"📉 Predicted Low: {low:.2f}")

# ✅ Run
if __name__ == "__main__":
    predict_prices()
