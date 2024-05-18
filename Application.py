import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model
model_path = 'water_consumption_lstm_model.h5'
model = load_model(model_path)

# Load the MinMaxScaler
scaler_path = 'scaler.pkl'  # Assuming you saved the scaler during model training
scaler = MinMaxScaler()
scaler = scaler.fit(pd.DataFrame({'Volume': [0]}))  # Dummy fit to avoid errors

# Function to preprocess user input
def preprocess_input(user_input, scaler):
    scaled_input = scaler.transform(user_input)
    return scaled_input

# Function to predict water consumption
def predict_water_consumption(user_input, model, scaler):
    # Preprocess the user input
    scaled_input = preprocess_input(user_input, scaler)
    
    # Predict water consumption
    prediction = model.predict(scaled_input)
    
    # Inverse transform the prediction to get the actual consumption value
    predicted_consumption = scaler.inverse_transform(prediction)
    
    return predicted_consumption

# Streamlit app
def main():
    st.title("Water Consumption Prediction")

    # User input for present consumption
    present_consumption = st.number_input("Enter your present water consumption (in cubic meters):", min_value=0.0)

    # Preprocess user input and make prediction
    user_input = pd.DataFrame({'Volume': [present_consumption]})
    predicted_future_consumption = predict_water_consumption(user_input, model, scaler)

    # Display the predicted future consumption
    st.subheader("Predicted Future Water Consumption:")
    st.write(predicted_future_consumption[0][0], "cubic meters")

if __name__ == "__main__":
    main()
