# Bike Rental Demand Prediction Using Neural Networks

## About the Project

This project predicts hourly bike rental demand using a neural network built with TensorFlow and Keras. It uses historical bike-sharing data, including weather conditions, temperature, humidity, season, and time-related features, to estimate the total number of bike rentals.

The project demonstrates an end-to-end machine learning workflow, from data exploration and preprocessing to model training, evaluation, and deployment through an interactive Streamlit web application.

## Objectives

- Analyze historical bike rental data to identify demand patterns.
- Build and train a neural network for rental demand prediction.
- Evaluate model performance against a baseline model.
- Develop an interactive web application for real-time predictions.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- TensorFlow and Keras
- Streamlit

## Features

- Exploratory data analysis and demand visualization
- Feature selection and preprocessing
- Neural network training and evaluation
- Saved model and feature scaler for inference
- Interactive Streamlit application for predicting bike rentals

## Model Performance

The trained neural network achieved a Mean Absolute Error (MAE) of **55.73 rentals**, compared with **168.34 rentals** for the baseline model on the evaluated dataset.

A lower MAE indicates that predictions are closer to the actual rental counts on average.

## How to Run the Project

1. Clone this repository.
2. Install the required Python libraries.
3. Ensure the trained model and scaler files are available in their expected locations.
4. Run the Streamlit application.

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow streamlit
streamlit run app.py
```

## Dataset

The project uses the Bike Sharing Dataset from the UCI Machine Learning Repository, specifically the hourly dataset (`hour.csv`).

Dataset source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## Future Improvements

- Experiment with additional neural network architectures.
- Improve feature engineering for time-based patterns.
- Compare performance with other machine learning algorithms.
- Deploy the application online.

## Author

Developed as a machine learning portfolio project to practice neural networks, model evaluation, and application deployment.
