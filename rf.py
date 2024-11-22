import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler


forest_path='./clean_data/forest.csv'
forest_df=pd.read_csv(forest_path)
X = forest_df[[f"tc_loss_ha_{year}" for year in range(2001, 2019)]]
y = forest_df[[f"tc_loss_ha_{year}" for year in range(2019, 2024)]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
multi_target_model = MultiOutputRegressor(gb_model)
multi_target_model.fit(X_train, y_train)
y_pred_multi = multi_target_model.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)

X_scaled = StandardScaler().fit_transform(X)
y_scaled = StandardScaler().fit_transform(y)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)

model = Sequential()
model.add(LSTM(units=128, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(units=64))
model.add(Dropout(0.25))
model.add(Dense(units=5))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
          callbacks=[early_stopping, reduce_lr])

y_pred_lstm = model.predict(X_test)
r2_lstm = r2_score(y_test, y_pred_lstm)

r2_scores = {'Random Forest': r2_rf, 'Gradient Boosting': r2_multi, 'LSTM': r2_lstm}

output_dir = 'lstm_evaluation'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('R² Score for Different Models')
plt.ylim([0, 1])

plt.savefig(os.path.join(output_dir, 'r2_score.png'))

print(f"R² Score for Random Forest: {r2_rf:.2f}")
print(f"R² Score for Gradient Boosting: {r2_multi:.2f}")
print(f"R² Score for LSTM: {r2_lstm:.2f}")
