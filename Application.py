import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

# Function to load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('water_consumption_lstm_model.h5')
    return model

# Define function to make predictions
def predict_consumption(model, previous_consumption):
    # Pad or repeat the single timestep value to create a sequence of length 10
    input_data = np.repeat(previous_consumption, 10).reshape((1, 10, 1))
    # Make prediction
    predicted_consumption = model.predict(input_data)
    return predicted_consumption[0][0]  # Assuming a scalar output

# Define function to predict 30 days of consumption
def predict_30_days(model, initial_consumption):
    predictions = []
    current_consumption = initial_consumption

    for _ in range(30):
        predicted_consumption = predict_consumption(model, current_consumption)
        predictions.append(predicted_consumption)
        current_consumption = predicted_consumption  # Use the latest prediction for the next input

    return predictions

# Streamlit app
def main():
    # set up the Streamlit app
    st.write("Final Exam: Deployment in Cloud")
    st.write("Name: Dhafny Buenafe and Mayah Catorce")
    st.write("Section: CPE32S3")
    st.write("Instructor: Engr. Roman Richard")

    st.title('Consumption Prediction App')

    # Load the pre-trained model
    model = load_model()

    # User input for previous consumption
    previous_consumption = st.number_input('Enter your previous consumption (cubic meters):', value=0.0)

    # Calculate price and predictions
    if st.button('Calculate Price'):
        # Calculation for a single input
        price = 1.48 * previous_consumption
        st.write(f'Entered consumption: {previous_consumption} cubic meters')
        st.write(f'Estimated price: {price:.2f} Php')

        # Generate 30-day predictions using the model
        daily_consumptions = predict_30_days(model, previous_consumption)
        daily_prices = [1.48 * consumption for consumption in daily_consumptions]
