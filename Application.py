import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('water_consumption_lstm_model.h5')

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler
    
def predict_water_consumption(input_data, scaler, time_steps):
    prepared_input = input_data.reshape(1, time_steps, -1)
    prediction = model.predict(prepared_input)
    predictions = scaler.inverse_transform(prediction)
    return predictions[0][0]

def calculate_water_cost(volume_m3, cost_per_m3):
    """
    Calculate the cost of water consumption.

    Args:
    - volume_m3 (float): The volume of water consumed in cubic meters (m^3).
    - cost_per_m3 (float): The cost per cubic meter of water in pesos.

    Returns:
    - float: The total cost of water consumption.
    """
    total_cost = volume_m3 * cost_per_m3
    return total_cost

def main():
    st.title("Water Consumption Prediction and Cost Calculator")
    
    # Input fields
    volume_m3 = st.number_input("Enter volume of water consumed (cubic meters)", min_value=0.0)
    time_steps = st.number_input("Enter time steps", min_value=1, value=10)
    cost_per_m3 = st.number_input("Enter cost per cubic meter of water (in pesos)", min_value=0.0)
    
    # Prediction and cost calculation
    if st.button("Predict and Calculate Cost"):
        # Dummy data for demonstration (replace this with actual input data)
        input_data = np.random.rand(time_steps)  # Sample input data
        
        # Prepare data
        scaled_input, scaler = prepare_data(input_data)
        
        # Predict water consumption
        predicted_consumption = predict_water_consumption(scaled_input, scaler, time_steps)
        
        # Calculate total cost
        total_cost = calculate_water_cost(volume_m3, cost_per_m3)
        
        # Display results
        st.subheader("Water Consumption Prediction:")
        st.write(f"Predicted water consumption: {predicted_consumption} cubic meters")
        
        st.subheader("Cost Calculation:")
        st.write(f"Total cost of water consumption: {total_cost} pesos")

if __name__ == "__main__":
    main()
