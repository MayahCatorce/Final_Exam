import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model for water consumption prediction
model = load_model('water_consumption_lstm_model.h5')

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def predict_water_consumption(input_data, scaler, time_steps):
    prepared_input = input_data.reshape(1, time_steps, -1)
    prediction = model.predict(prepared_input)
    predicted_consumption = scaler.inverse_transform(prediction)
    return predicted_consumption[0][0]

def main():
    st.title("Water Consumption Prediction")

    # User input for volume of water consumed
    volume_m3 = st.number_input("Enter volume of water consumed (cubic meters)", min_value=0.0)

    # Prepare data
    input_data = np.array([volume_m3])  # Assume user input is a single value
    scaled_input, scaler = prepare_data(input_data)

    # Predict water consumption
    predicted_consumption = predict_water_consumption(scaled_input, scaler, time_steps=1)

    # Display prediction
    st.subheader("Predicted Water Consumption:")
    st.write(f"{predicted_consumption:.2f} cubic meters")

if __name__ == "__main__":
    main()
