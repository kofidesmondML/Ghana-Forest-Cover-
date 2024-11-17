import pandas as pd

license_path = './data/report_license_04112024.xlsx'
forest_path = './clean_data/forest.csv'

license = pd.read_excel(license_path)
forest_df = pd.read_csv(forest_path)

def process_license_data(license):
    try:
        license_df = license[license['Type'].str.contains("small scale mining", case=False, na=False)]
        license_df = license_df[license_df['Status'] == 'Active License']
    except KeyError as e:
        print(f"Error: Missing column {e} in the DataFrame.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during filtering: {e}")
        return None
    
    try:
        split_df = license_df['Regions'].str.split(r"\n", n=1, expand=True)
        if split_df.shape[1] == 2:
            license_df[['Region', 'District']] = split_df
        else:
            print("Error: Some rows do not contain the delimiter or have unexpected formatting.")
            return None
    except KeyError as e:
        print(f"Error: Missing 'Regions' column in the DataFrame.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during splitting the 'Regions' column: {e}")
        return None

    try:
        license_df['District'] = license_df['District'].str.split(r"\n").apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        license_df['District'] = license_df['District'].str.split(',')
        license_df = license_df.explode('District')
        license_df['District'] = license_df['District'].str.strip()
    except KeyError as e:
        print(f"Error: Missing 'District' column in the DataFrame.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during the processing of the 'District' column: {e}")
        return None

    return license_df

license_data = process_license_data(license)
if license_data is not None:
    print(license_data.head())

district_license_count = license_data.groupby('District').size().reset_index(name='Number of Licenses')
district_license_count.to_csv('./clean_data/district_license_count.csv', index=False)

def get_number_of_licenses(district):
    try:
        if (district_license_count['District'] == district).any():
            return district_license_count.loc[district_license_count['District'] == district, 'Number of Licenses'].iloc[0]
        else:
            return None
    except Exception as e:
        return None
district_license_count['District'] = district_license_count['District'].str.replace(r'\s*(District|Region)\s*', '', regex=True)
forest_df['Number of Licenses'] = forest_df['subnational2'].apply(get_number_of_licenses)
print(forest_df['Number of Licenses'].nunique())

