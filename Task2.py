# Unemployment Analysis in India - CodeAlpha Internship Task

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("Unemployment in India.csv")  # <- Ensure file is in same folder

# 2. Clean column names
df.columns = [col.strip() for col in df.columns]

# 3. Display basic info
print("Dataset Info:\n", df.info())
print("\nMissing values:\n", df.isnull().sum())

# 4. Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# 5. Overview
print("\nSample Data:\n", df.head())

# 6. Visualizations

# Line plot - National unemployment rate over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', label='Unemployment Rate')
plt.title("Unemployment Rate Over Time in India")
plt.xlabel("Date")
plt.ylabel("Estimated Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap - State-wise average unemployment
pivot_state = df.pivot_table(values='Estimated Unemployment Rate (%)', index='Region', aggfunc='mean')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_state, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Average Unemployment Rate by Region")
plt.tight_layout()
plt.show()

# Monthly trend (COVID-19 impact)
df['Month'] = df['Date'].dt.to_period('M')
monthly_trend = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(12, 6))
monthly_trend.plot(marker='o')
plt.title("Monthly Average Unemployment Rate (COVID Impact Highlighted)")
plt.axvspan(pd.Period('2020-03'), pd.Period('2021-03'), color='red', alpha=0.2, label='COVID-19 Period')
plt.xlabel("Month")
plt.ylabel("Estimated Unemployment Rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
