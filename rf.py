import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os

file_path = './clean_data/forest.csv'
data = pd.read_csv(file_path)

columns = [f"tc_loss_ha_{year}" for year in range(2001, 2024)]
data = data[columns].T
data.index = range(2001, 2024)
data.columns = [f"region_{i}" for i in range(data.shape[1])]

train_years = 18
test_years = 5
X_train = data.iloc[:train_years, :]
X_test = data.iloc[train_years:, :]
y_train = X_train.mean(axis=1)
y_test = X_test.mean(axis=1)

def create_lagged_features(data, n_lags=3):
    lagged_data = []
    for i in range(n_lags, len(data)):
        lagged_data.append(data.iloc[i-n_lags:i, :].values.flatten())
    return np.array(lagged_data)

lags = 3
X_train_lagged = create_lagged_features(X_train, lags)
X_test_lagged = create_lagged_features(X_test, lags)
y_train_lagged = y_train[lags:]
y_test_lagged = y_test[lags:]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_lagged, y_train_lagged)
rf_preds = rf_model.predict(X_test_lagged)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_lagged, y_train_lagged)
xgb_preds = xgb_model.predict(X_test_lagged)

lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train_lagged, y_train_lagged)
lgb_preds = lgb_model.predict(X_test_lagged)

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    return mae, mse, r2

print("Evaluating Models:")
rf_results = evaluate(y_test_lagged, rf_preds, "Random Forest")
xgb_results = evaluate(y_test_lagged, xgb_preds, "XGBoost")
lgb_results = evaluate(y_test_lagged, lgb_preds, "LightGBM")

def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2019, 2024)[lags:], y_true, label="Actual", marker='o')
    plt.plot(range(2019, 2024)[lags:], y_pred, label=f"{model_name} Predictions", marker='x')
    plt.xlabel("Year")
    plt.ylabel("Tree Cover Loss (ha)")
    plt.title(f"{model_name} Predictions vs Actual")
    plt.legend()

output_dir = './predictions_evaluation'
os.makedirs(output_dir, exist_ok=True)

plot_predictions(y_test_lagged, rf_preds, "Random Forest")
plt.savefig(f"{output_dir}/rf_predictions.png")
plt.close()

plot_predictions(y_test_lagged, xgb_preds, "XGBoost")
plt.savefig(f"{output_dir}/xgb_predictions.png")
plt.close()

plot_predictions(y_test_lagged, lgb_preds, "LightGBM")
plt.savefig(f"{output_dir}/lgb_predictions.png")
plt.close()

print("Plots saved to 'predictions_evaluation' folder.")
