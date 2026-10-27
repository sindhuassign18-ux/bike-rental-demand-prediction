
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("hour.csv")

# Calculate average rentals for each weekday
weekday_demand = df.groupby("weekday")["cnt"].mean()

# Display the values
print("Average rentals by weekday:")
print(weekday_demand)

# Plot the results
plt.figure(figsize=(8, 5))
plt.bar(weekday_demand.index, weekday_demand.values)

plt.title("Average Bike Rentals by Weekday")
plt.xlabel("Weekday (0 = Sunday)")
plt.ylabel("Average Rentals per Hour")
plt.xticks(range(7))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
