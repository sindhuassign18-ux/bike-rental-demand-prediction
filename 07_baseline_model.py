
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and sort the dataset
df = pd.read_csv("hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

# Select features and target
features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]
X = df[features]
y = df["cnt"]

# Split chronologically by whole dates
unique_dates = df["dteday"].drop_duplicates()

train_dates, remaining_dates = train_test_split(
    unique_dates, test_size=0.30, shuffle=False
)
val_dates, test_dates = train_test_split(
    remaining_dates, test_size=0.50, shuffle=False
)

# Select rows for each set
train_mask = df["dteday"].isin(train_dates)
val_mask = df["dteday"].isin(val_dates)

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]

# Create and train the baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

# Predict validation-set demand
y_pred = baseline.predict(X_val)

# Evaluate performance
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("Baseline Model: Mean Predictor")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Mean rental count predicted:", round(y_pred[0], 2))
print("Validation MAE:", round(mae, 2))
print("Validation RMSE:", round(rmse, 2))
