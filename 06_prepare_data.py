
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("hour.csv")

# Ensure the observations are in chronological order
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

# Select input features
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]

# Separate inputs (X) and target (y)
X = df[features]
y = df["cnt"]


# Get unique dates in chronological order
unique_dates = df["dteday"].drop_duplicates()

# Split into training dates (70%) and remaining dates (30%)
train_dates, remaining_dates = train_test_split(
    unique_dates,
    test_size=0.30,
    shuffle=False
)

# Split remaining dates into validation (15%) and test (15%)
val_dates, test_dates = train_test_split(
    remaining_dates,
    test_size=0.50,
    shuffle=False
)


# Select rows belonging to each date group
train_mask = df["dteday"].isin(train_dates)
val_mask = df["dteday"].isin(val_dates)
test_mask = df["dteday"].isin(test_dates)

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

# Display dataset sizes
print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# Display date ranges
print("\nTraining dates:", train_dates.iloc[0].date(),
      "to", train_dates.iloc[-1].date())

print("Validation dates:", val_dates.iloc[0].date(),
      "to", val_dates.iloc[-1].date())

print("Test dates:", test_dates.iloc[0].date(),
      "to", test_dates.iloc[-1].date())

# Verify that the date groups do not overlap
print("\nTrain/validation overlap:",
      not set(train_dates).isdisjoint(set(val_dates)))

print("Validation/test overlap:",
      not set(val_dates).isdisjoint(set(test_dates)))

print("\nTraining input shape:", X_train.shape)
print("Training target shape:", y_train.shape)
