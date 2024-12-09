# Ghana-Forest-Cover-
## Forest Change Detection and Predictive Modeling

This project focuses on analyzing tree cover loss data, detecting change points, and implementing predictive models using different machine learning algorithms. The goal is to understand forest loss patterns over time and provide forecasts for future tree cover loss using models like SARIMA, Random Forest, Gradient Boosting, and LSTM.

## Project Overview

The project involves several key tasks:

1. **Tree Cover Loss Analysis:** 
   - Time series analysis and forecasting using SARIMA (Seasonal ARIMA) for predicting future tree cover loss based on historical data.
   
2. **Predictive Modeling:**
   - Implementing various machine learning models including Random Forest, Gradient Boosting, and LSTM to predict future tree cover loss.
   - The models are trained using historical tree cover loss data from 2001 to 2023 and tested to predict data for the years 2019 to 2023.
   
3. **Change Point Detection:**
   - Using the CUSUM (Cumulative Sum) method to detect significant changes in tree cover loss over time for different subnational districts.

4. **Evaluation and Visualization:**
   - Evaluating the models' performance based on metrics such as R², Mean Absolute Error (MAE), and Mean Squared Error (MSE).
   - Visualizing results such as the tree cover loss over time, forecasting, and change point detections for each district.

## Requirements

To run this project, you will need the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `statsmodels`
- `scikit-learn`
- `keras`
- `tensorflow`

### `requirements.txt`

The following dependencies are required for the project:

