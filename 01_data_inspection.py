import pandas as pd
df = pd.read_csv("hour.csv")
print(df.head())


# Check the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display all column names
print("\nColumn names:")
print(df.columns.tolist())

# Check data types and non-null counts
print("\nDataset information:")
df.info()

# Count missing values in each column
print("\nMissing values:")
print(df.isnull().sum())
print("\nUnique season values:")
print(sorted(df["season"].unique()))

print("\nNumber of records per season:")
print(df["season"].value_counts().sort_index())