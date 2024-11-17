import numpy as np
import pandas as pd

forest_path='./clean_data/forest.csv'

forest_df=pd.read_csv(forest_path)


# Example CUSUM function for detecting change points
def cusum_detection(series, threshold):
    mean_value = np.mean(series)
    cumsum = np.cumsum(series - mean_value)
    change_points = np.where(np.abs(cumsum) > threshold)[0]
    return change_points

# Applying CUSUM to each district's forest loss data
for district in forest_df['subnational2'].unique():
    series = forest_df[forest_df['subnational2'] == district].iloc[:, 9:32].values.flatten()
    change_points = cusum_detection(series, threshold=100)
    print(f"Change points for {district}: {change_points}")
