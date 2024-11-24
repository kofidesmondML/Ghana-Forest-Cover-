import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import seaborn as sns
import folium 


forest_path = './clean_data/forest.csv'
save_folder = './eda_results'

# Read the dataset
df = pd.read_csv(forest_path)

# Function to plot tree cover loss for a specific constituency
def plot_tree_cover_loss(constituency_name):
    filtered_df = df[df['subnational2'] == constituency_name]
    print(f'Plotting for tree cover loss for {constituency_name}')
    
    if filtered_df.empty:
        print("No data found for this constituency.")
        return
    
    # Remove unnecessary columns
    filtered_df = filtered_df.drop(columns=['country', 'subnational1', 'subnational2', 'threshold', 
                                             'area_ha', 'extent_2000_ha', 'extent_2010_ha', 
                                             'gain_2000-2020_ha'])
    
    years = list(range(2001, 2024, 1))
    tree_cover_loss = filtered_df.values.tolist()

    # Plot and save
    plt.figure(figsize=(15, 10))
    plt.style.use('ggplot')
    plt.plot(years, tree_cover_loss[0], marker='o', markerfacecolor='red', color='b')
    plt.title(f'Tree Cover Loss for {constituency_name}')
    plt.xlabel('Year')
    plt.ylabel('Tree Cover Loss')
    plt.xticks(years, rotation=45)
    plt.grid(True)
    
    save_path = os.path.join(save_folder, f"{constituency_name}_tree_cover_loss.png")
    plt.savefig(save_path)
    plt.close()

# Plot for each constituency
for constituency_name in df['subnational2'].to_list():
    plot_tree_cover_loss(constituency_name)

# Plot total tree cover loss over the years
def plot_total_tree_cover_loss():
    years = list(range(2001, 2024))
    total_loss_per_year = []
    
    for year in years:
        column_name = f'tc_loss_ha_{year}'
        if column_name in df.columns:
            total_loss_per_year.append(df[column_name].sum())
    
    plt.figure(figsize=(15, 10))
    plt.plot(years, total_loss_per_year, marker='o', color='blue', markerfacecolor='red', markeredgecolor='black')
    plt.title('Total Tree Cover Loss 2001 - 2023', fontsize=20)
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Total Tree Cover Loss (ha)', fontsize=15)
    plt.grid(True)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, 'total_tree_cover_loss_2001_2023.png')
    plt.savefig(save_path)
    plt.close()

plot_total_tree_cover_loss()

# Plot distribution of tree cover gain (2000-2020)
def plot_tree_cover_gain_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gain_2000-2020_ha'], bins=10, kde=True, color='green', edgecolor='black')
    plt.title('Distribution of Tree Cover Gain 2000-2020 Across Constituencies', fontsize=16)
    plt.xlabel('Tree Cover Gain (ha)', fontsize=14)
    plt.ylabel('Number of Constituencies', fontsize=14)
    plt.grid(True)
    
    save_path = os.path.join(save_folder, 'tree_cover_gain_distribution.png')
    plt.savefig(save_path)
    plt.close()

plot_tree_cover_gain_distribution()

# Plot cumulative net loss for each constituency
def plot_cumulative_net_loss(df, constituency_names):
    for constituency_name in constituency_names:
        constituency_data = df[df['subnational2'] == constituency_name]
        
        if constituency_data.empty:
            print(f"No data available for {constituency_name}.")
            continue
        
        extent_2000_ha = constituency_data['extent_2000_ha'].values[0]
        years = list(range(2001, 2024)) 
        net_loss_cols = [f'tc_loss_ha_{year}' for year in years]
        net_losses = constituency_data[net_loss_cols].values.flatten()
        cumulative_net_losses = net_losses.cumsum()
        cumulative_percentage_net_losses = (cumulative_net_losses / extent_2000_ha) * 100
        
        plt.figure(figsize=(10, 6))
        plt.plot(years, cumulative_percentage_net_losses, marker='o', linestyle='-', color='r', label=f"Cumulative Net Loss % in {constituency_name}")
        plt.title(f"Cumulative Percentage of Tree Cover Net Loss Relative to 2000 in {constituency_name}")
        plt.xlabel("Year")
        plt.ylabel("Cumulative Percentage Net Loss (%)")
        plt.xticks(years, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(save_folder, f'cumulative_net_loss_{constituency_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

constituency_list = df['subnational2'].to_list()
plot_cumulative_net_loss(df, constituency_list)

# Plot total tree cover loss by region
def plot_tree_cover_loss_by_region(region):
    region_df = grouped_df[grouped_df['subnational1'] == region]
    
    if region_df.empty:
        print(f"No data found for the region: {region}")
        return
    
    plt.figure(figsize=(12, 6))
    plt.bar(region_df['subnational2'], region_df['total_tc_loss'], color='blue')
    plt.xticks(rotation=90)
    plt.title(f"Total Tree Cover Loss for Constituencies in {region}")
    plt.xlabel("Constituency")
    plt.ylabel("Total Tree Cover Loss (ha)")
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f"tree_cover_loss_{region}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Group data by region and plot for each region
df['total_tc_loss'] = df.loc[:, 'tc_loss_ha_2001':'tc_loss_ha_2023'].sum(axis=1)
grouped_df = df.groupby(['subnational1', 'subnational2'])['total_tc_loss'].sum().reset_index()

regions = df['subnational1'].unique()
for region_name in regions:
    plot_tree_cover_loss_by_region(region_name)

gold_constituencies=['Obuasi Municipal', 'Tarkwa-Nsuaem','Mpohor Wassa East','Asutifi', 'Birim North', 'Bibiani Anhwiaso Bekwai','Upper Denkyira', 'Amansie West', 'Atiwa','Fanteakwa']
gold_producing = df[df['subnational2'].isin(gold_constituencies)]
non_gold_producing = df[~df['subnational2'].isin(gold_constituencies)]
print(gold_producing.describe())
gold_yearly_loss = gold_producing.loc[:, 'tc_loss_ha_2001':'tc_loss_ha_2023'].sum()
non_gold_yearly_loss = non_gold_producing.loc[:, 'tc_loss_ha_2001':'tc_loss_ha_2023'].sum()

# Plotting the data
plt.figure(figsize=(15, 12))
years = range(2001, 2024)
plt.plot(years, gold_yearly_loss, label='Gold Producing', marker='o', color='gold')
plt.plot(years, non_gold_yearly_loss, label='Non-Gold Producing', marker='o', color='green')
plt.xticks(years, rotation=45)
plt.title('Yearly Tree Cover Loss (2001-2023)')
plt.xlabel('Year')
plt.ylabel('Average Tree Cover Loss (ha)')
plt.legend()
plt.grid(True)

# Save the plot to a file
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist
save_path = os.path.join(save_folder, 'yearly_tree_cover_loss_gold_vs_non_gold.png')
plt.savefig(save_path, dpi=300)
plt.close()

constituency_coordinates = {
    'Obuasi Municipal': '6.2137° N, 1.6955° W',
    'Tarkwa-Nsuaem': '5.1817° N, 2.0273° W',
    'Mpohor Wassa East': '5.1333° N, 1.65° W',
    'Asutifi': '6.9577° N, 2.4209° W',
    'Birim North': '6.3855° N, 1.0012° W',
    'Bibiani Anhwiaso Bekwai': '6.2752° N, 2.2630° W',
    'Upper Denkyira': '5.8539° N, 1.7150° W',
    'Amansie West': '6.4469° N, 1.8709° W',
    'Atiwa': '6.4286° N, 0.6583° W',
    'Fanteakwa': '6.4502° N, 0.3372° W'
}

def create_constituency_map(constituency_coordinates):
    os.makedirs('folium_maps', exist_ok=True)
    for constituency, coords in constituency_coordinates.items():
        m = folium.Map(location=[6.2137, -1.6955], zoom_start=7)
        lat, lon = coords.split(', ')
        if 'S' in lat:
            lat = -float(lat.replace("° S", "").strip())
        else:
            lat = float(lat.replace("° N", "").strip())
        if 'W' in lon:
            lon = -float(lon.replace("° W", "").strip())
        else:
            lon = float(lon.replace("° E", "").strip())
        folium.Marker([lat, lon], popup=constituency).add_to(m)
        save_path = f'./folium_maps/{constituency.replace(" ", "_")}_map.html'
        m.save(save_path)

create_constituency_map(constituency_coordinates)


