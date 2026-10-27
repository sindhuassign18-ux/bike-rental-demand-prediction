
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hour.csv")

# Scatter plot: temperature vs. rentals
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["cnt"], alpha=0.15)

plt.title("Temperature vs. Bike Rental Demand")
plt.xlabel("Normalized Temperature")
plt.ylabel("Total Rentals per Hour")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate correlation
correlation = df["temp"].corr(df["cnt"])
print("Temperature-rental correlation:", correlation)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hour.csv")

# Scatter plot: temperature vs. rentals
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["cnt"], alpha=0.15)

plt.title("Temperature vs. Bike Rental Demand")
plt.xlabel("Normalized Temperature")
plt.ylabel("Total Rentals per Hour")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate correlation
correlation = df["temp"].corr(df["cnt"])
print("Temperature-rental correlation:", correlation)
