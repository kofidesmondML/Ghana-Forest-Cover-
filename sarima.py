import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('ggplot')
import numpy as np
import os 
os.makedirs('./sarima_evaluation', exist_ok=True)

forest_df = pd.read_csv('./clean_data/forest.csv')
tc_loss_columns = [col for col in forest_df.columns if 'tc_loss_ha_' in col]
tc_loss_data = forest_df[tc_loss_columns].mean(axis=0)
tc_loss_data.index = range(2001, 2024)

plt.figure(figsize=(12, 6))
plt.plot(tc_loss_data.index, tc_loss_data, marker='o', linestyle='-', color='blue', label='Observed')
plt.title('Tree Cover Loss (Ha) Over Time')
plt.xlabel('Year')
plt.ylabel('Tree Cover Loss (Ha)')
plt.xticks(tc_loss_data.index, rotation=45)
plt.grid(True)
plt.legend()
plt.savefig('./sarima_evaluation/tree_cover_loss_over_time.png')

result = adfuller(tc_loss_data)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] > 0.05:
    print("The time series is non-stationary. Differencing is required.")
    tc_loss_data_diff = tc_loss_data.diff().dropna()
else:
    print("The time series is stationary.")
    tc_loss_data_diff = tc_loss_data

plot_acf(tc_loss_data_diff)
plt.savefig('./sarima_evaluation/acf_plot.png')

plot_pacf(tc_loss_data_diff)
plt.savefig('./sarima_evaluation/pacf_plot.png')

best_aic = np.inf
best_order = None
best_seasonal_order = None
best_model = None

for p in range(0, 3):
    for d in range(0, 2):
        for q in range(0, 3):
            for P in range(0, 2):
                for D in range(0, 2):
                    for Q in range(0, 2):
                        try:
                            model = SARIMAX(tc_loss_data, order=(p, d, q), 
                                            seasonal_order=(P, D, Q, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                            model_fit = model.fit(disp=False)
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_order = (p, d, q)
                                best_seasonal_order = (P, D, Q, 12)
                                best_model = model_fit
                        except:
                            continue

print(f"Best SARIMA Order: {best_order}")
print(f"Best Seasonal Order: {best_seasonal_order}")
print(best_model.summary())

forecast_steps = 10
forecast = best_model.forecast(steps=forecast_steps)
forecast_index = range(2024, 2024 + forecast_steps)

plt.figure(figsize=(12, 6))
plt.plot(tc_loss_data.index, tc_loss_data, marker='o', linestyle='-', color='blue', label='Observed')
plt.plot(forecast_index, forecast, marker='s', linestyle='--', color='red', label='Forecast')
plt.title('Tree Cover Loss Forecast')
plt.xlabel('Year')
plt.ylabel('Tree Cover Loss (Ha)')
plt.xticks(list(tc_loss_data.index) + list(forecast_index), rotation=45)
plt.grid(True)
plt.legend()
plt.savefig('./sarima_evaluation/tree_cover_loss_forecast.png')
