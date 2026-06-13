import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")

print("Dataset Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# ---------------------------
# Graph 1: Unemployment Rate by Region
# ---------------------------

plt.figure(figsize=(12,6))

region_data = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

sns.barplot(
    x=region_data.values,
    y=region_data.index
)

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# ---------------------------
# Graph 2: Urban vs Rural
# ---------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x='Area',
    y='Estimated Unemployment Rate (%)'
)

plt.title("Urban vs Rural Unemployment")
plt.tight_layout()
plt.show()

# ---------------------------
# Graph 3: Monthly Trend
# ---------------------------

monthly = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(12,6))

monthly.plot()

plt.title("Unemployment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------
# Key Statistics
# ---------------------------

print("\nAverage Unemployment Rate:")
print(df['Estimated Unemployment Rate (%)'].mean())

print("\nHighest Unemployment State:")
print(region_data.idxmax())

print("\nHighest Unemployment Rate:")
print(region_data.max())