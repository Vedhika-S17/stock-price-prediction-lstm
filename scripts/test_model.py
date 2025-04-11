import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import os
from IPython.display import Image, display

# Load data
df = pd.read_csv("data/AAPL_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
data = df[["High", "Low"]].values

# Load scaler
scaler = joblib.load("models/scaler.save")
scaled_data = scaler.transform(data)

# Sequence setup
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_len = 60
X_all, y_all = create_sequences(scaled_data, seq_len)

# Split (same as training)
total_len = len(X_all)
train_len = int(total_len * 0.65)
val_len = int(total_len * 0.10)
test_start = train_len + val_len

X_test = X_all[test_start:]
y_test = y_all[test_start:]

# Load model
model = load_model("models/lstm_model.keras")

# Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Extract High and Low separately
y_test_high, y_test_low = y_test_actual[:, 0], y_test_actual[:, 1]
y_pred_high, y_pred_low = y_pred[:, 0], y_pred[:, 1]

# Evaluation metrics
def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    print(f"\n🔹 {label} Price Prediction:")
    print(f"   ▪️ RMSE  : {rmse:.4f}")
    print(f"   ▪️ MAE   : {mae:.4f}")
    print(f"   ▪️ MAPE  : {mape:.2f}%")
    print(f"   ▪️ R²    : {r2:.4f}")
    print(f"   ▪️ Explained Variance: {evs:.4f}")

    # Simple inference
    if r2 > 0.6:
        print(f"✅ {label} price prediction is reliable.")
    elif r2 > 0.2:
        print(f"⚠️ {label} price prediction is moderately useful, but can be improved.")
    else:
        print(f"❌ {label} price prediction is currently not reliable.\n")

evaluate(y_test_high, y_pred_high, "High")
evaluate(y_test_low, y_pred_low, "Low")

# Plot predictions
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(y_test_high, label="Actual High", color="skyblue")
plt.plot(y_pred_high, label="Predicted High", color="darkblue")
plt.title("High Price Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test_low, label="Actual Low", color="lightcoral")
plt.plot(y_pred_low, label="Predicted Low", color="darkred")
plt.title("Low Price Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Save and display the figure
os.makedirs("results", exist_ok=True)
plot_path = "results/prediction_plot.png"
plt.tight_layout()
plt.savefig(plot_path)

# Display the image (for Jupyter/Colab)
display(Image(filename=plot_path))
