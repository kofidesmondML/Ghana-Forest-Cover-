import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load data
forest_path = './clean_data/forest.csv'
forest_df = pd.read_csv(forest_path)

# Split features and labels
X = forest_df[[f"tc_loss_ha_{year}" for year in range(2001, 2019)]]
y = forest_df[[f"tc_loss_ha_{year}" for year in range(2019, 2024)]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
multi_target_model = MultiOutputRegressor(gb_model)
multi_target_model.fit(X_train, y_train)
y_pred_multi = multi_target_model.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
mae_multi = mean_absolute_error(y_test, y_pred_multi)

# LSTM Model
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)

def combined_loss(y_true, y_pred, alpha=0.5):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return alpha * mse_loss + (1 - alpha) * mae_loss

model = Sequential()
model.add(LSTM(units=128, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.25))
model.add(Dense(units=64))
model.add(Dropout(0.25))
model.add(Dense(units=5))
model.compile(optimizer='adam', loss=combined_loss)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[early_stopping, reduce_lr])

y_pred_lstm = model.predict(X_test)
 
y_pred_lstm_orig = scaler_y.inverse_transform(y_pred_lstm)
y_test_orig = scaler_y.inverse_transform(y_test)

r2_lstm = r2_score(y_test_orig, y_pred_lstm_orig)
mse_lstm = mean_squared_error(y_test_orig, y_pred_lstm_orig)
mae_lstm = mean_absolute_error(y_test_orig, y_pred_lstm_orig)

# Evaluation Metrics
r2_scores = {'Random Forest': r2_rf, 'Gradient Boosting': r2_multi, 'LSTM': r2_lstm}
mse_scores = {'Random Forest': mse_rf, 'Gradient Boosting': mse_multi, 'LSTM': mse_lstm}
mae_scores = {'Random Forest': mae_rf, 'Gradient Boosting': mae_multi, 'LSTM': mae_lstm}

# Plot Results
output_dir = 'lstm_evaluation'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.ylabel('R² Score', fontsize=14, fontweight='bold')
plt.title('R² Score for Models', fontsize=16, fontweight='bold')
plt.ylim([0, 1])
for i, v in enumerate(r2_scores.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
r2_plot_path = os.path.join(output_dir, 'r2_score.png')
plt.savefig(r2_plot_path)

plt.figure(figsize=(8, 6))
plt.bar(mse_scores.keys(), mse_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
plt.title('Mean Squared Error for Models', fontsize=16, fontweight='bold')
for i, v in enumerate(mse_scores.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
mse_plot_path = os.path.join(output_dir, 'mse_score.png')
plt.savefig(mse_plot_path)

plt.figure(figsize=(8, 6))
plt.bar(mae_scores.keys(), mae_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.ylabel('Mean Absolute Error', fontsize=14, fontweight='bold')
plt.title('Mean Absolute Error for Models', fontsize=16, fontweight='bold')
for i, v in enumerate(mae_scores.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
mae_plot_path = os.path.join(output_dir, 'mae_score.png')
plt.savefig(mae_plot_path)

# Print Results
print("R² Scores:", r2_scores)
print("Mean Squared Error Scores:", mse_scores)
print("Mean Absolute Error Scores:", mae_scores)
