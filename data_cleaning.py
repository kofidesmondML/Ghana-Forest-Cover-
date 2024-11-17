import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

forest_path='./data/GHA.xlsx'

forest_df=pd.read_excel(forest_path, sheet_name='Subnational 2 tree cover loss')
readme=pd.read_excel(forest_path, sheet_name='Read_Me')
forest_df = forest_df[forest_df['threshold']==75]
print(forest_df)

forest_df.to_csv('./clean_data/forest.csv', index=False)