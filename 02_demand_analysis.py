import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv("hour.csv")

hourly_demand = df.groupby('hr')['cnt'].mean()

plt.figure(figsize=(10, 5))
plt.title('Average bike rentals per hour')
plt.plot(hourly_demand.index, hourly_demand.values)
plt.xlabel('Hour')
plt.ylabel('Average Demand')
plt.title('Hourly Demand Analysis')
plt.show()