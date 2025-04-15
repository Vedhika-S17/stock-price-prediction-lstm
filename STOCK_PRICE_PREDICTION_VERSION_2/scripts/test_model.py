import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Load scaled data and scalers
data = np.load('data/scaled_data.npy')  # The pre-scaled data (5 features)
y_high = np.load('data/y_high.npy')  # The high price targets
y_low = np.load('data/y_low.npy')  # The low price targets
feature_scaler = joblib.load('data/feature_scaler.pkl')  # The feature scaler for the 5 features
high_scaler = joblib.load('data/high_scaler.pkl')  # High price scaler
low_scaler = joblib.load('data/low_scaler.pkl')  # Low price scaler

# Train-test split (80-20, no shuffle)
_, X_test, _, y_test_high = train_test_split(data, y_high, test_size=0.2, shuffle=False)
_, _, _, y_test_low = train_test_split(data, y_low, test_size=0.2, shuffle=False)

# Load models
model_high = load_model('models/model_high_price.keras')
model_low = load_model('models/model_low_price.keras')

# Predict on test data
pred_high = model_high.predict(X_test).flatten()
pred_low = model_low.predict(X_test).flatten()

# Inverse transform predictions and actual values
pred_high_inv = high_scaler.inverse_transform(pred_high.reshape(-1, 1)).flatten()
pred_low_inv = low_scaler.inverse_transform(pred_low.reshape(-1, 1)).flatten()
y_test_high_inv = high_scaler.inverse_transform(y_test_high.reshape(-1, 1)).flatten()
y_test_low_inv = low_scaler.inverse_transform(y_test_low.reshape(-1, 1)).flatten()

# Evaluation function
def evaluate(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    acc = 100 - (mae / np.mean(y_true) * 100)
    print(f"🔹 {label} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Accuracy Est: {acc:.2f}%")
    return mse, rmse, mae, r2, acc

# Evaluate both models
print("\n📊 Evaluation Metrics:")
evaluate(y_test_high_inv, pred_high_inv, "High Price")
evaluate(y_test_low_inv, pred_low_inv, "Low Price")

# Plot predictions vs actual values
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(y_test_high_inv, label='Actual High', color='blue')
plt.plot(pred_high_inv, label='Predicted High', color='orange')
plt.title('High Price: Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test_low_inv, label='Actual Low', color='green')
plt.plot(pred_low_inv, label='Predicted Low', color='red')
plt.title('Low Price: Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.savefig('plots/high_low_prediction.png')
plt.show()

# Print interpretation of metrics
print("\n📘 Interpretation:")
print("- MSE: Average squared error (lower is better).")
print("- RMSE: Square root of MSE, same unit as price.")
print("- MAE: Average of absolute errors.")
print("- R² Score: Variance explained by the model.")
print("- Accuracy Est: Based on deviation from mean.")
