import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

forest_path = './clean_data/forest.csv'
forest_df = pd.read_csv(forest_path)
print(forest_df.shape)

output_folder = './change_detection'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def cusum_detection(series, threshold):
    mean_value = np.mean(series)
    cumsum = np.cumsum(series - mean_value)
    change_points = np.where(np.abs(cumsum) > threshold)[0]
    return change_points

years = np.arange(2001, 2024)
#print(len(years))

for district in forest_df['subnational2'].unique():
    series = forest_df[forest_df['subnational2'] == district].iloc[:, 8:31].values.flatten()
    print(len(series))

    if len(series) != len(years):
        print(f"Skipping {district} due to mismatched data length: {len(series)} vs {len(years)}")
        continue

    change_points = cusum_detection(series, threshold=50)

    if len(change_points) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(years, series, label='Forest Loss Data', color='blue', marker='o')
        plt.scatter(years[change_points], series[change_points], color='red', label='Change Points', zorder=5)
        plt.title(f"Change Points for {district}")
        plt.xlabel("Year")
        plt.ylabel("Forest Loss (ha)")
        plt.xticks(years, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        file_path = os.path.join(output_folder, f"{district}_change_points.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
