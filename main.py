import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Load the data
df = pd.read_csv('2022_Green_Taxi_Trip_Data_20240621.csv')
print("HEAD!")
print(df.head())
print("INFO!")
print(df.info())

# Convert pickup and dropoff times to datetime type
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

# Calculate trip duration in minutes
df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

# Remove outliers (trips longer than 3 hours and shorter than 1 minute)
df = df[(df['trip_duration'] >= 1) & (df['trip_duration'] <= 180)]

print(df['trip_duration'].describe())

# Round to two decimal places
df['trip_duration'] = df['trip_duration'].round(2)

print(df[['lpep_pickup_datetime', 'lpep_dropoff_datetime','trip_duration' ]].head())

# Visualization of trip duration distribution
plt.figure(figsize=(10,6))
sns.histplot(df['trip_duration'], bins=100, edgecolor='black')
plt.title('Distribution of Trip Distances (0-100 miles)')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Frequency')
plt.xlim(0, 100)
plt.show()

df_nonzero = df[df['trip_distance'] > 0]
df_morethanone = df[df['trip_distance'] > 1]
df_filtered = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]

# Visualization of trip distance vs duration
plt.figure(figsize=(10,6))
sns.scatterplot(x='trip_distance', y='trip_duration', data=df_filtered, alpha=0.1)
plt.title('Trip Distance vs Duration (Excluding Zero-Distance Trips)')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Trip Duration (minutes)')
plt.show()

print(df['trip_distance'].describe())

#TIME SERIES ANALYSIS

# Analyze trips by hour of day
df['hour'] = df['lpep_pickup_datetime'].dt.hour
hourly_trips = df.groupby('hour').size()

plt.figure(figsize=(12, 6))
hourly_trips.plot(kind='bar')
plt.title('Number of Trips by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()

# EMISSION SAVINGS FOR EV TAXI'S

average_co2_gas = 411 #grams per mile (g/mi)
average_co2_electric = 0

total_distance = df['trip_distance'].sum() #miles

total_emissions_gas = average_co2_gas * total_distance
total_emissions_electric = average_co2_electric * total_distance

# emissions savings
emissions_savings = total_emissions_gas - total_emissions_electric #grams
emissions_savings_kg = emissions_savings / 1000

print(f"Total Emissions Savings: {emissions_savings_kg:.2f} kg")
#28,873,785 KG