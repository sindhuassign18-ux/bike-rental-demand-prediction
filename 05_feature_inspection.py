
import pandas as pd

df = pd.read_csv("hour.csv")

# Check dataset dimensions
print("Dataset shape:", df.shape)

# Check missing values
print("\nMissing values:", df.isnull().sum().sum())

# Display all column names
print("\nColumns:")
print(df.columns.tolist())

# Inspect the target variable
print("\nRental count statistics:")
print(df["cnt"].describe())

# Verify how total rentals are calculated
print(
    "\nDoes cnt equal casual + registered for every row?",
    (df["cnt"] == df["casual"] + df["registered"]).all()
)
