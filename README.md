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


To install the dependencies, run the following command:

```bash
pip install -r requirements.txt

## Data

The dataset used for this project, `forest.csv`, contains information about tree cover loss (in hectares) over the years, from 2001 to 2023. The columns include:

- `tc_loss_ha_<year>`: Tree cover loss (ha) for each year.
- `subnational2`: Geographic district information.

Ensure that the dataset is available in the `./clean_data/` directory before running the scripts.

## Model Training

The project includes three machine learning models for predicting tree cover loss:

1. **Random Forest Regressor**: A robust ensemble model for regression tasks.
2. **Gradient Boosting Regressor**: Another ensemble model that builds trees sequentially.
3. **LSTM (Long Short-Term Memory)**: A deep learning model suitable for time-series forecasting.

### To run the predictive modeling:

1. **Load Data**: The data is read from `forest.csv` and processed into features (`X`) and target labels (`y`).
2. **Train and Test Split**: The dataset is split into training and testing sets.
3. **Train Models**: Each model is trained using the training set.
4. **Evaluate Models**: The models performance is evaluated using R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).
5. **Save Results**: The evaluation results and plots are saved in the `./lstm_evaluation` directory.

### To run the script for training and evaluation, run:

```bash
python rf.py

## Results

After running the model training, the results will be saved in the `./lstm_evaluation/` folder:

- **R², MSE, and MAE bar plots** for each model (`r2_score.png`, `mse_score.png`, `mae_score.png`).
- A **text file** (`evaluation_metrics.txt`) containing the evaluation metrics.

## Change Detection

Change detection is performed using the CUSUM method to identify significant changes in tree cover loss for each district. The method accumulates the difference between each data point and the mean, and flags when the accumulated sum exceeds a specified threshold.

### To run the change detection:

Run the script for change point detection:

```bash
python change_detection.py


