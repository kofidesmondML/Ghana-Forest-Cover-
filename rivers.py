import pandas as pd 


river_path='./data/Aqueduct40_baseline_annual_y2023m07d05.csv'

river_df=pd.read_csv(river_path)
river_df=river_df[river_df['name_0']=='Ghana']

print(river_df.head())

river_df.to_csv('./clean_data/river_aqueduct.csv')