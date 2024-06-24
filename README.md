# 2022-green-taxi-emissions-analysis

These records are generated from the trip record submissions made by green taxi Technology Service Providers (TSPs). Each row represents a single trip in a green taxi. The trip records include fields capturing pick-up and drop-off dates/times, pick-up and drop-off taxi zone locations, trip distances, itemized fares, rate types, payment types, and driver-reported passenger counts.

# Taxi Trip Data Analysis

This project involves analyzing taxi trip data from New York City to gain insights into trip durations, distances, and the potential environmental impact of switching to electric vehicles (EVs).

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Analysis and Visualizations](#analysis-and-visualizations)
  - [Data Preprocessing](#data-preprocessing)
  - [Trip Duration Analysis](#trip-duration-analysis)
  - [Trip Distance vs Duration](#trip-distance-vs-duration)
  - [Time Series Analysis](#time-series-analysis)
  - [Emissions Savings with EVs](#emissions-savings-with-evs)
- [Requirements](#requirements)
- [Usage](#usage)

## Introduction
This project analyzes NYC taxi trip data to:
- Visualize trip durations and distances.
- Analyze trip patterns over time.
- Estimate emissions savings if taxis were switched to electric vehicles.

## Dataset
The dataset used is `2022_Green_Taxi_Trip_Data_20240621.csv`, which contains detailed information about taxi trips, including pickup and dropoff times, locations, distances, and fares.

## Analysis and Visualizations

### Data Preprocessing
1. **Load the Data**: Read the dataset and display the first few rows.
2. **Convert Datetime Columns**: Convert pickup and dropoff times to datetime format.
3. **Calculate Trip Duration**: Compute the duration of each trip in minutes.
4. **Remove Outliers**: Filter out trips shorter than 1 minute and longer than 3 hours.

```python
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
print(df.head())
print(df.info())

# Convert pickup and dropoff times to datetime type
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

# Calculate trip duration in minutes
df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

# Remove outliers (trips longer than 3 hours and shorter than 1 minute)
df = df[(df['trip_duration'] >= 1) & (df['trip_duration'] <= 180)]

print(df['trip_duration'].describe())
```

### Trip Duration Analysis
Visualize the distribution of trip durations to understand the typical duration of taxi rides.

```python
# Round to two decimal places
df['trip_duration'] = df['trip_duration'].round(2)

print(df[['lpep_pickup_datetime', 'lpep_dropoff_datetime','trip_duration' ]].head())

# Visualization of trip duration distribution
plt.figure(figsize=(10,6))
sns.histplot(df['trip_duration'], bins=100, edgecolor='black')
plt.title('Distribution of Trip Durations')
plt.xlabel('Trip Duration (minutes)')
plt.ylabel('Frequency')
plt.xlim(0, 180)
plt.show()
```
![distribution_trip_distances](https://github.com/haydenbalsys/2022-green-taxi-emissions-analysis/assets/74757315/501136aa-6fc5-4f37-a9e0-e88d15b58534)

### Trip Distance vs Duration
Examine the relationship between trip distance and duration.

```python
df_filtered = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]

# Visualization of trip distance vs duration
plt.figure(figsize=(10,6))
sns.scatterplot(x='trip_distance', y='trip_duration', data=df_filtered, alpha=0.1)
plt.title('Trip Distance vs Duration (Excluding Zero-Distance Trips)')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Trip Duration (minutes)')
plt.show()

print(df['trip_distance'].describe())
```
![trip_distance_vs_duration](https://github.com/haydenbalsys/2022-green-taxi-emissions-analysis/assets/74757315/c23ba361-422d-4eb9-8269-b6139ff47c96)

### Time Series Analysis
Analyze the number of trips by hour of the day to identify peak times.

```python
# Analyze trips by hour of day
df['hour'] = df['lpep_pickup_datetime'].dt.hour
hourly_trips = df.groupby('hour').size()

plt.figure(figsize=(12, 6))
hourly_trips.plot(kind='bar')
plt.title('Number of Trips by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()
```
![number_trips_by_hour_of_day](https://github.com/haydenbalsys/2022-green-taxi-emissions-analysis/assets/74757315/3673f3dd-d70c-4330-92a4-7de91159a82b)

### Emissions Savings with EVs
Estimate the potential reduction in CO2 emissions if all taxis were electric vehicles.

```python
# Emission Savings for EV Taxis
average_co2_gas = 411  # grams per mile (g/mi)
average_co2_electric = 0

total_distance = df['trip_distance'].sum()  # miles

total_emissions_gas = average_co2_gas * total_distance
total_emissions_electric = average_co2_electric * total_distance

# Emissions savings
emissions_savings = total_emissions_gas - total_emissions_electric  # grams
emissions_savings_kg = emissions_savings / 1000

print(f"Total Emissions Savings: {emissions_savings_kg:.2f} kg")
```

# 28,873,785 KG

### Requirements
- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- folium
- scikit-learn
- plotly
