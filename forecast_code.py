# -*- coding: utf-8 -*-
"""Climate Change Forecasting Model"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load datasets
co2_data = pd.read_csv('co2_mm_mlo.csv')
temperature_data = pd.read_csv('temperature.csv')
global_data = pd.read_csv('global.csv')
ocean_temp_data = pd.read_csv('ocean_temp_data.csv')
sunspot_flux_data = pd.read_csv('flux_sunspot.csv')
forest_data = pd.read_csv('forest_size.csv')

# Plot CO2 concentration
annual_co2 = co2_data.groupby('year')['average'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.plot(annual_co2['year'], annual_co2['average'], marker='o', color='black', linestyle='-')
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO₂ Average Concentration (ppm)', fontsize=12)
plt.title('Global CO₂ Concentration Changes (Annual Average)', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Merge temperature data
total_temp = pd.merge(temperature_data, global_data, left_on='year', right_on='year')

# Clean temperature data
for col in ['D-N', 'DJF', 'MAM', 'JJA', 'SON']:
    temperature_data[col] = pd.to_numeric(temperature_data[col], errors='coerce')

temperature_data = temperature_data[(temperature_data[['D-N', 'DJF', 'MAM', 'JJA', 'SON']] >= -1).all(axis=1)]
temperature_data = temperature_data[temperature_data['year'] >= 1980]
temperature_data['Annual Mean'] = temperature_data[['D-N', 'DJF', 'MAM', 'JJA', 'SON']].mean(axis=1)

# Plot temperature data
plt.figure(figsize=(12, 8))
years = temperature_data['year']
plt.bar(years, temperature_data['D-N'], label='D-N', color='navy')
plt.bar(years, temperature_data['DJF'], bottom=temperature_data['D-N'], label='DJF', color='blue')
plt.bar(years, temperature_data['MAM'], bottom=temperature_data['D-N'] + temperature_data['DJF'], label='MAM', color='skyblue')
plt.bar(years, temperature_data['JJA'], bottom=temperature_data['D-N'] + temperature_data['DJF'] + temperature_data['MAM'], label='JJA', color='lightgreen')
plt.bar(years, temperature_data['SON'], bottom=temperature_data['D-N'] + temperature_data['DJF'] + temperature_data['MAM'] + temperature_data['JJA'], label='SON', color='orange')
plt.plot(years, temperature_data['Annual Mean'], color='black', marker='o', label='Annual Mean')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Annual Temperature Components and Mean (Starting from 1980)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot ocean temperature histogram
ocean_temp_data.set_index('year', inplace=True)
plt.figure(figsize=(12, 6))
plt.hist(ocean_temp_data['Heat content anomaly'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Ocean Temperature Data: Heat Content Anomaly')
plt.xlabel('Heat Content Anomaly Values')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Plot sunspot flux data
sunspot_flux_data.set_index('year', inplace=True)
sunspot_flux_data = sunspot_flux_data[(sunspot_flux_data.index >= 1955) & (sunspot_flux_data.index <= 2023)]
plt.figure(figsize=(12, 6))
plt.plot(sunspot_flux_data.index, sunspot_flux_data['smoothed_ssn'], label='Smoothed Solar Sunspot Number', color='blue')
plt.plot(sunspot_flux_data.index, sunspot_flux_data['smoothed_swpc_ssn'], label='Smoothed SWPC SSN', color='orange')
plt.plot(sunspot_flux_data.index, sunspot_flux_data['smoothed_f10.7'], label='Smoothed F10.7', color='green')
plt.title('Sunspot Flux Data (1955 to 2023)')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.xlim([1955, 2023])
plt.show()

# Merge all datasets
merged_df = pd.merge(co2_data, total_temp, on='year', how='inner')
merged_df = pd.merge(merged_df, ocean_temp_data, on='year', how='inner')
merged_df = pd.merge(merged_df, sunspot_flux_data, on='year', how='inner')
merged_df = pd.merge(merged_df, forest_data, on='year', how='inner')

# Clean merged data
for col in merged_df.columns:
    merged_df[col] = merged_df[col].replace(',', '', regex=True)
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

correlation_matrix = merged_df.corr()
print(correlation_matrix)

# Prepare features and target
X = merged_df[['average', 'deseasonalized', 'D-N', 'DJF', 'MAM', 'JJA', 'SON',
               'smoothed_ssn', 'smoothed_swpc_ssn', 'smoothed_f10.7',
               'forest size in million hectares']]
y = merged_df['Heat content anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ClimateStack Model Implementation (Random Forest + XGBoost Stacking)
base_model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)
meta_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

stacking_model = StackingRegressor(
    estimators=[('random_forest', base_model)],
    final_estimator=meta_model,
    cv=3
)

stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)

# Calculate ClimateStack metrics
r2 = r2_score(y_test, y_pred)
r2_percentage = r2 * 100
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R-squared (Test):", r2)
print("Accuracy (R-squared %):", r2_percentage)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Plot ClimateStack forecast
start_year = 1958
historical_years = len(y_train)
years_historical = np.arange(start_year, start_year + historical_years)
years_future = np.arange(2024, 2024 + len(y_pred))
years = np.concatenate([years_historical, years_future])
values = np.concatenate([y_train, y_pred])

df = pd.DataFrame({"Year": years, "Value": values})
df["Rolling_Avg"] = df["Value"].rolling(window=5, min_periods=1).mean()

plt.figure(figsize=(12, 8))
plt.plot(df["Year"][:historical_years], df["Rolling_Avg"][:historical_years],
         label="Historical (5-Year Avg)", color="black")
plt.axvline(x=2023, color="black", linestyle="--", label="Prediction Start")
plt.text(2024, df["Rolling_Avg"].max(), "Prediction Starts Here", color="black")
plt.fill_betweenx(y=[df["Rolling_Avg"].min(), df["Rolling_Avg"].max()],
                  x1=2023, x2=2040, color='lightgreen', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Quantitative value of climate change (5-Year Average)")
plt.title("1958–2040 Climate Change Quantification Curve (5-Year Rolling Average) of ClimateStack")
plt.xticks(np.arange(start_year, start_year + len(values), 5))
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlim(start_year, 2040)
plt.ylim(bottom=0)
plt.show()
