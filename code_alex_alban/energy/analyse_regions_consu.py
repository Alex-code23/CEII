import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

def clean_column_names(df):
    """Cleans up the column names of the DataFrame."""
    new_columns = {
        'Date - Heure': 'datetime',
        'Code INSEE région': 'region_code',
        'Région': 'region',
        'Consommation brute gaz (MW PCS 0°C) - GRTgaz': 'gas_grtgaz_mw',
        'Statut - GRTgaz': 'status_grtgaz',
        'Consommation brute gaz (MW PCS 0°C) - Teréga': 'gas_terega_mw',
        'Statut - Teréga': 'status_terega',
        'Consommation brute gaz totale (MW PCS 0°C)': 'gas_total_mw',
        'Consommation brute électricité (MW) - RTE': 'electricity_rte_mw',
        'Statut - RTE': 'status_rte',
        'Consommation brute totale (MW)': 'total_consumption_mw'
    }
    df = df.rename(columns=new_columns)
    return df

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the energy consumption data."""
    try:
        # Load data with semicolon delimiter
        df = pd.read_csv(filepath, sep=';', low_memory=False)

        # Clean column names
        df = clean_column_names(df)

        # Convert datetime column and set as index
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.set_index('datetime')

        # Select and convert numeric columns
        consumption_cols = [
            'gas_grtgaz_mw', 'gas_terega_mw', 'gas_total_mw',
            'electricity_rte_mw', 'total_consumption_mw'
        ]
        for col in consumption_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill missing consumption values with 0, assuming no data means no consumption from that source
        df[consumption_cols] = df[consumption_cols].fillna(0)

        # Drop rows where total consumption is 0 as they are likely data errors
        df = df[df['total_consumption_mw'] > 0]
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.day_name()
        df['month'] = df.index.month
        
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_consumption_over_time(df, output_dir):
    """Plots total energy consumption over time."""
    plt.figure(figsize=(18, 8))
    df['total_consumption_mw'].resample('D').mean().plot()
    plt.title('Average Daily Total Energy Consumption Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Consumption (MW)')
    plt.savefig(os.path.join(output_dir, '01_avg_daily_consumption_over_time.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_consumption_by_region(df, output_dir):
    """Plots total energy consumption by region."""
    plt.figure(figsize=(18, 10))
    region_consumption = df.groupby('region')['total_consumption_mw'].sum().sort_values(ascending=False)
    sns.barplot(x=region_consumption.values, y=region_consumption.index, hue=region_consumption.index, palette='viridis', legend=False)
    plt.title('Total Energy Consumption by Region')
    plt.xlabel('Total Consumption (MW)')
    plt.ylabel('Region')
    plt.savefig(os.path.join(output_dir, '02_total_consumption_by_region.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_daily_consumption_pattern(df, output_dir):
    """Plots the average consumption pattern over a day."""
    plt.figure(figsize=(14, 7))
    df.groupby('hour')['total_consumption_mw'].mean().plot(kind='bar')
    plt.title('Average Hourly Energy Consumption Pattern')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Consumption (MW)')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(output_dir, '03_avg_hourly_consumption_pattern.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_weekly_consumption_pattern(df, output_dir):
    """Plots the average consumption pattern over a week."""
    plt.figure(figsize=(14, 7))
    weekly_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df.groupby('day_of_week')['total_consumption_mw'].mean().reindex(weekly_order).plot(kind='bar')
    plt.title('Average Daily Energy Consumption Pattern (Weekly)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Consumption (MW)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, '04_avg_weekly_consumption_pattern.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_gas_vs_electricity(df, output_dir):
    """Plots the comparison of gas and electricity consumption."""
    plt.figure(figsize=(18, 8))
    df_resampled = df[['gas_total_mw', 'electricity_rte_mw']].resample('W').mean()
    df_resampled.plot(ax=plt.gca())
    plt.title('Average Weekly Gas vs. Electricity Consumption')
    plt.xlabel('Date')
    plt.ylabel('Average Consumption (MW)')
    plt.legend(['Total Gas', 'Electricity'])
    plt.savefig(os.path.join(output_dir, '05_avg_weekly_gas_vs_electricity.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_consumption_heatmap(df, output_dir):
    """Plots a heatmap of consumption by hour and day of the week."""
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(values='total_consumption_mw', index=df.index.hour, columns=df.index.dayofweek)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table.columns = [days[i] for i in pivot_table.columns]
    pivot_table = pivot_table[days] # Order columns
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=False)
    plt.title('Heatmap of Average Energy Consumption by Hour and Day of Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Hour of the Day')
    plt.savefig(os.path.join(output_dir, '06_consumption_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_daily_profiles_by_region(df, output_dir, top_n=6):
    """
    Plots average daily consumption profiles for the top N consuming regions.
    """
    plt.figure(figsize=(18, 10))
    
    # Find top N regions by total consumption
    top_regions = df.groupby('region')['total_consumption_mw'].sum().nlargest(top_n).index
    df_top = df[df['region'].isin(top_regions)]
    
    # Pivot data to get hourly consumption for each top region
    daily_profiles = df_top.groupby(['hour', 'region'])['total_consumption_mw'].mean().unstack() / 1000 # in GW
    
    daily_profiles.plot(ax=plt.gca(), colormap='viridis', linewidth=2.5)
    
    plt.title(f'Profil de Consommation Journalier Moyen pour les {top_n} plus Grandes Régions (GW)', fontsize=16)
    plt.xlabel('Heure de la journée', fontsize=12)
    plt.ylabel('Consommation Moyenne (GW)', fontsize=12)
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Région')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_daily_profiles_by_region.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_monthly_consumption_by_region(df, output_dir):
    """Plots average monthly consumption trends for each region."""
    plt.figure(figsize=(18, 10))
    monthly_regional = df.groupby(['month', 'region'])['total_consumption_mw'].mean().unstack() / 1000 # in GW
    
    monthly_regional.plot(ax=plt.gca(), colormap='tab20', linewidth=2)
    
    plt.title('Profil de Consommation Mensuel Moyen par Région (GW)', fontsize=16)
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Consommation Moyenne (GW)', fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
    plt.legend(title='Région', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_monthly_consumption_by_region.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_gas_electricity_ratio_by_region(df, output_dir):
    """Plots the ratio of total gas to electricity consumption for each region."""
    regional_sums = df.groupby('region')[['gas_total_mw', 'electricity_rte_mw']].sum()
    regional_sums['ratio_gas_elec'] = regional_sums['gas_total_mw'] / regional_sums['electricity_rte_mw']
    regional_sums = regional_sums.sort_values('ratio_gas_elec', ascending=False)
    
    plt.figure(figsize=(18, 10))
    sns.barplot(x=regional_sums['ratio_gas_elec'], y=regional_sums.index, hue=regional_sums.index, palette='coolwarm', legend=False)
    plt.title('Ratio de la Consommation Totale Gaz / Électricité par Région', fontsize=16)
    plt.xlabel('Ratio (Consommation Gaz / Consommation Électricité)', fontsize=12)
    plt.ylabel('Région', fontsize=12)
    plt.axvline(1, color='black', linestyle='--', label='Ratio = 1 (Consommation égale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_gas_electricity_ratio_by_region.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the analysis."""
    filepath = r'C:\Users\Alexander\Documents\GitHub\CEII\data\raw\energy\ODRE_consommation_quotidienne_brute_regionale.csv'
    output_dir = r'c:\Users\Alexander\Documents\GitHub\CEII\results\analyse_regions_consu'
    df = load_and_preprocess_data(filepath)

    os.makedirs(output_dir, exist_ok=True)

    if df is not None:
        print("Data loaded and preprocessed successfully.")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        print("\nBasic statistics of consumption data:")
        print(df[['gas_total_mw', 'electricity_rte_mw', 'total_consumption_mw']].describe())

        # --- Generate Plots ---
        print("\nGenerating plots...")
        
        # 1. Total consumption over time
        plot_consumption_over_time(df, output_dir)
        
        # 2. Total consumption by region
        plot_consumption_by_region(df, output_dir)
        
        # 3. Gas vs. Electricity
        plot_gas_vs_electricity(df, output_dir)
        
        # 4. Daily consumption pattern
        plot_daily_consumption_pattern(df, output_dir)
        
        # 5. Weekly consumption pattern
        plot_weekly_consumption_pattern(df, output_dir)

        # 6. Heatmap of consumption
        plot_consumption_heatmap(df, output_dir)

        # --- Nouveaux graphiques de comparaison régionale ---
        print("\nGenerating regional comparison plots...")

        # 7. Profils journaliers par région (Top 6)
        plot_daily_profiles_by_region(df, output_dir, top_n=6)
        # 8. Tendances mensuelles par région
        plot_monthly_consumption_by_region(df, output_dir)
        # 9. Ratio Gaz/Électricité par région
        plot_gas_electricity_ratio_by_region(df, output_dir)

        print(f"\nAnalysis complete. Plots saved in '{output_dir}'")

if __name__ == '__main__':
    main()
