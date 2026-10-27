
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================================================
# 1. REPRODUCIBILITY
# ==================================================
tf.keras.utils.set_random_seed(42)

# ==================================================
# 2. LOAD AND PREPARE DATA
# ==================================================
df = pd.read_csv("hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])

df = df.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)

# Create cyclical time features
df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)

# ==================================================
# 3. SELECT FEATURES AND TARGET
# ==================================================
features = [
    "season", "yr", "mnth",
    "hour_sin", "hour_cos",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed"
]

X = df[features]
y = df["cnt"]

# ==================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ==================================================
unique_dates = df["dteday"].drop_duplicates()

train_dates, remaining_dates = train_test_split(
    unique_dates,
    test_size=0.30,
    shuffle=False
)

val_dates, test_dates = train_test_split(
    remaining_dates,
    test_size=0.50,
    shuffle=False
)

train_mask = df["dteday"].isin(train_dates)
val_mask = df["dteday"].isin(val_dates)
test_mask = df["dteday"].isin(test_dates)

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

print("=== Data Preparation ===")
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# ==================================================
# 5. NORMALIZE FEATURES
# Fit only on training data to prevent leakage
# ==================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 6. BUILD NEURAL NETWORK
# ==================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(features),)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# ==================================================
# 7. COMPILE MODEL
# ==================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

# ==================================================
# 8. EARLY STOPPING
# ==================================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

# ==================================================
# 9. TRAIN MODEL
# ==================================================
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining completed.")

best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_loss = min(history.history["val_loss"])

print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 2))

# ==================================================
# 10. VALIDATION EVALUATION
# ==================================================
y_val_pred = model.predict(
    X_val_scaled, verbose=0
).flatten()

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(
    mean_squared_error(y_val, y_val_pred)
)

print("\n=== Validation Results ===")
print("Validation MAE:", round(val_mae, 2))
print("Validation RMSE:", round(val_rmse, 2))

# ==================================================
# 11. PLOT LEARNING CURVES
# ==================================================
epochs_ran = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs_ran,
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs_ran,
    history.history["val_loss"],
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================
# 12. BASELINE MODEL
# ==================================================
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train_scaled, y_train)

baseline_val_pred = baseline.predict(X_val_scaled)
baseline_test_pred = baseline.predict(X_test_scaled)

baseline_val_mae = mean_absolute_error(
    y_val, baseline_val_pred
)
baseline_val_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_val_pred)
)

# ==================================================
# 13. FINAL TEST EVALUATION
# Do not use test results for model tuning
# ==================================================
y_test_pred = model.predict(
    X_test_scaled, verbose=0
).flatten()

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

baseline_test_mae = mean_absolute_error(
    y_test, baseline_test_pred
)
baseline_test_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_test_pred)
)

print("\n=== FAIR VALIDATION COMPARISON ===")
print("Baseline MAE:", round(baseline_val_mae, 2))
print("Neural Network MAE:", round(val_mae, 2))
print("Baseline RMSE:", round(baseline_val_rmse, 2))
print("Neural Network RMSE:", round(val_rmse, 2))

print("\n=== FINAL TEST COMPARISON ===")
print("Baseline Test MAE:", round(baseline_test_mae, 2))
print("Neural Network Test MAE:", round(test_mae, 2))
print("Baseline Test RMSE:", round(baseline_test_rmse, 2))
print("Neural Network Test RMSE:", round(test_rmse, 2))

# ==================================================
# 14. SAMPLE PREDICTIONS
# ==================================================
print("\n=== FIRST 10 TEST PREDICTIONS ===")

for actual, predicted in zip(
    y_test.iloc[:10], y_test_pred[:10]
):
    print(
        f"Actual: {actual:4d} | "
        f"Predicted: {predicted:.2f}"
    )

# ==================================================
# 15. ERROR ANALYSIS
# ==================================================
error_df = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": y_test_pred,
    "hour": df.loc[test_mask, "hr"].to_numpy()
})

error_df["absolute_error"] = (
    error_df["actual"] - error_df["predicted"]
).abs()

error_df["signed_error"] = (
    error_df["predicted"] - error_df["actual"]
)

print("\n=== 10 LARGEST PREDICTION ERRORS ===")
print(
    error_df.sort_values(
        "absolute_error", ascending=False
    ).head(10)[
        ["actual", "predicted", "absolute_error", "hour"]
    ].round(2).to_string(index=False)
)

# Demand-level error analysis
error_df["demand_level"] = pd.cut(
    error_df["actual"],
    bins=[-1, 100, 300, 600, float("inf")],
    labels=["Low", "Moderate", "High", "Very High"]
)

print("\n=== MAE BY DEMAND LEVEL ===")
print(
    error_df.groupby(
        "demand_level", observed=False
    )["absolute_error"].mean().round(2)
)

print("\n=== PREDICTION BIAS BY DEMAND LEVEL ===")
print(
    error_df.groupby(
        "demand_level", observed=False
    )["signed_error"].mean().round(2)
)

# Error analysis by hour
hour_errors = error_df.groupby("hour").agg(
    MAE=("absolute_error", "mean"),
    Mean_Bias=("signed_error", "mean")
).round(2)

print("\n=== PREDICTION ERRORS BY HOUR ===")
print(hour_errors)

plt.figure(figsize=(10, 5))
plt.bar(hour_errors.index, hour_errors["MAE"])
plt.xlabel("Hour of Day")
plt.ylabel("Mean Absolute Error")
plt.title("Prediction Error by Hour")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Actual versus predicted plot
plt.figure(figsize=(7, 6))

plt.scatter(
    error_df["actual"],
    error_df["predicted"],
    alpha=0.35
)

max_value = max(
    error_df["actual"].max(),
    error_df["predicted"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.title("Actual vs Predicted Bike Rentals")
plt.tight_layout()
plt.show()

print("\nAll evaluation and error analysis completed.")

# Save the trained model and scaler
os.makedirs("models", exist_ok=True)

model.save("models/bike_rental_model.keras")
joblib.dump(scaler, "models/feature_scaler.pkl")

print("\nModel and scaler saved successfully.")




