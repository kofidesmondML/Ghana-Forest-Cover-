import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import traceback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def prepare_data(district_data, sequence_length=10):
    try:
        losses = district_data.loc[:, 'tc_loss_ha_2001':'tc_loss_ha_2023'].values.flatten()
        if len(losses) <= sequence_length:
            print(f"Not enough data for sequence preparation in district: {district_data['subnational2'].iloc[0]}")
            return None, None, None
        
        scaler = MinMaxScaler()
        losses_scaled = scaler.fit_transform(losses.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(losses_scaled) - sequence_length):
            X.append(losses_scaled[i:i + sequence_length])
            y.append(losses_scaled[i + sequence_length])
        
        X = np.array(X).reshape((-1, sequence_length, 1))
        y = np.array(y)
        return X, y, scaler
    except Exception as e:
        print(f"Error preparing data for district {district_data['subnational2'].iloc[0]}: {e}")
        return None, None, None

def build_lstm_model(input_shape):
    try:
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("LSTM model built successfully")
        return model
    except Exception as e:
        print(f"Error building LSTM model: {e}")
        return None

def predict_future(model, last_sequence, scaler, n_predictions=5):
    try:
        assert last_sequence.ndim == 3 and last_sequence.shape[0] == 1, \
            "last_sequence must have shape (1, sequence_length, 1)"
        predictions = []
        for _ in range(n_predictions):
            next_pred = model.predict(last_sequence, verbose=0)
            assert next_pred.shape == (1, 1), f"Unexpected shape for next_pred: {next_pred.shape}"
            predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
            assert last_sequence.shape[1] == last_sequence.shape[1], \
                f"last_sequence has unexpected shape: {last_sequence.shape}"
        assert hasattr(scaler, 'scale_'), "Scaler is not fitted"
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"Error predicting future values: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_district(district_data, sequence_length=10):
    try:
        print(f"Processing district: {district_data['subnational2'].iloc[0]} with sequence length {sequence_length}")
        X, y, scaler = prepare_data(district_data, sequence_length)
        if X is None or y is None:
            return None
        
        #    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        model = build_lstm_model((X.shape[1], 1))
        if model is None:
            return None
        
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        last_sequence = X[-1].reshape(1, sequence_length, 1)
        predictions = predict_future(model, last_sequence, scaler)
        return predictions
    except Exception as e:
        print(f"Error processing district {district_data['subnational2'].iloc[0]}: {e}")
        return None
def reshape_and_save(lstm_df, output_path='./clean_data/predictions.csv'):
    try:
        reshaped_df = lstm_df.transpose()
        reshaped_df = reshaped_df.reset_index()
        reshaped_df = reshaped_df.rename(columns={'index': 'District'})
        reshaped_df.columns = ['District', 2024, 2025, 2026, 2027, 2028]
        reshaped_df.to_csv(output_path, index=False)
        return reshaped_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def main():
    forest_path = './clean_data/forest.csv'
    forest_df = load_data(forest_path)

    if forest_df is None:
        return

    predictions_dict = {}
    districts = forest_df['subnational2'].unique()

    for district in districts:
        print(district)
        try:
            district_data = forest_df[forest_df['subnational2'] == district]
            predictions = process_district(district_data)
            if predictions is not None:
                predictions_dict[district] = predictions
            else:
                print(f"Prediction not available for district: {district}")
        except Exception as e:
            print(f"Error processing district {district}: {e}")
    
    for district, predictions in predictions_dict.items():
        print(f"Predictions for {district}: {predictions}")
    predictions_df=pd.DataFrame(predictions_dict)
    predictions_df.to_csv('./clean_data/lstm_predictions.csv', index=False)
    lstm_df=pd.read_csv('./clean_data/lstm_predictions.csv')
    final_df=reshape_and_save(lstm_df)
    print(final_df.head())
if __name__ == "__main__":
    main()
