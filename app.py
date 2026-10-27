import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="SPS",
    layout="centered"
)

# Page heading
st.title("Bike Rental Demand Predictor")
st.write(
    "Estimate hourly bike rental demand using a trained neural network."
)

st.info(
    "Enter the conditions for the hour you want to predict. "
    "The model will estimate the total number of bike rentals."
)

# Check saved files
MODEL_PATH = Path("models/bike_rental_model.keras")
SCALER_PATH = Path("models/feature_scaler.pkl")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model files not found. Check your models folder.")
    st.stop()

# Load saved model and scaler
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

st.success("Trained model loaded successfully!")

st.subheader("Enter prediction details")

# Input controls — interface only for now
col1, col2 = st.columns(2)

with col1:
    hour = st.selectbox(
        "Hour of day",
        options=list(range(24)),
        index=8,
        format_func=lambda h: f"{h:02d}:00"
    )

    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Autumn",
            4: "Winter"
        }[x]
    )

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=5
    )

    year = st.selectbox(
        "Year",
        options=[0, 1],
        format_func=lambda x: "2011" if x == 0 else "2012"
    )

with col2:
    weather = st.selectbox(
        "Weather",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear or partly cloudy",
            2: "Mist or cloudy",
            3: "Light rain or snow",
            4: "Heavy rain or snow"
        }[x]
    )

    
    temperature_c = st.slider(
        "Temperature (°C)",
        min_value=-8.0,
        max_value=39.0,
        value=20.0,
        step=0.5
    )

    feels_like_c = st.slider(
        "Feels-like temperature (°C)",
        min_value=-16.0,
        max_value=50.0,
        value=20.0,
        step=0.5
    )

    humidity_pct = st.slider(
        "Relative humidity (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1
    )

    windspeed_original = st.slider(
        "Wind speed (dataset scale)",
        min_value=0.0,
        max_value=67.0,
        value=13.4,
        step=0.5
    )


st.caption(
    "These temperature, feels-like, humidity, and wind values use "
    "the dataset's normalized scale, not degrees Celsius or percentages."
)

st.subheader("Additional conditions")

col3, col4 = st.columns(2)

with col3:
    holiday = st.selectbox("Holiday?", options=[0, 1],
                           format_func=lambda x: "Yes" if x else "No")

    workingday = st.selectbox("Working day?", options=[0, 1],
                              format_func=lambda x: "Yes" if x else "No")

with col4:
    weekday = st.selectbox(
        "Day of week",
        options=list(range(7)),
        format_func=lambda x: [
            "Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday"
        ][x]
    )


st.write("Click below to get your bike rental demand prediction.")

if st.button("🚲 Predict Bike Rentals", type="primary",
              use_container_width=True):

    # Prepare the 13 features in the exact training order
    feature_cols = [
        "season", "yr", "mnth", "hour_sin", "hour_cos",
        "holiday", "weekday", "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed"
    ]

    input_data = {
        "season": season,
        "yr": year,
        "mnth": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weather,
        "temp": temperature_c / 41,
        "atemp": feels_like_c / 50,
        "hum": humidity_pct / 100,
        "windspeed": windspeed_original / 67
    }

    # Arrange inputs in the same order used during training
    input_df = pd.DataFrame([input_data], columns=feature_cols)

    # Apply the saved scaler
    scaled_input = scaler.transform(input_df)

    # Predict using the trained neural network
    prediction = model.predict(scaled_input, verbose=0)[0][0]

    # Display the result
    predicted_rentals = max(0, round(float(prediction)))

    st.success("Prediction completed!")
    st.metric(
        label="Estimated bike rentals for this hour",
        value=f"{predicted_rentals:,} bikes"
    )

    st.caption(
        "This is a model estimate based on the conditions you entered, "
        "not a guarantee of actual demand."
    )
