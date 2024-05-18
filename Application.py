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

# Streamlit app
def main():
    # set up the Streamlit app
    st.write("Final Exam: Deployment in Cloud")
    st.write("Name: Dhafny Buenafe and Mayah Catorce")
    st.write("Section: CPE32S3")
    st.write("Instructor: Engr. Roman Richard")

    # Load the pre-trained model
    model = load_model()

    st.title('Consumption Prediction App')

    # User input for previous consumption
    previous_consumption = st.number_input('Enter your previous consumption (cubic meters):', value=0.0)

    # Predict consumption
    if st.button('Predict Consumption'):
        # Make predictions
        predicted_consumption = predict_consumption(model, previous_consumption)
        price = 14.8 * predicted_consumption  # Calculate price
        st.write(f'Predicted consumption: {predicted_consumption} cubic meters')
        st.write(f'Estimated price: ${price:.2f}')

if __name__ == "__main__":
    main()
