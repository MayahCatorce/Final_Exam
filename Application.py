import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_path = 'water_consumption_lstm_model.h5'
model = load_model(model_path)

# Load your dataset for reference
df = pd.read_csv('water_consumption.csv')

# Fit the scaler on your training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df)  # Assuming df contains your training data

# Function to preprocess user input
def preprocess_input(user_input, scaler):
    scaled_input = scaler.transform(np.array(user_input).reshape(-1, 1))
    return scaled_input

def main():
    st.title("Water Consumption Prediction")

    # User input for water consumption in cubic meters
    user_input = st.number_input("Enter your water consumption (cubic meters):", min_value=0.0)

    # Preprocess the user input
    scaled_input = preprocess_input(user_input, scaler)

    # Predict water consumption
    prediction = model.predict(np.reshape(scaled_input, (1, scaled_input.shape[0], 1)))

    # Inverse transform the prediction to get the actual consumption value
    predicted_consumption = scaler.inverse_transform(prediction.reshape(-1, 1))

    # Calculate the cost
    cost_per_cubic_meter = 14.8  # Cost per cubic meter in pesos
    total_cost = predicted_consumption * cost_per_cubic_meter

    st.subheader("Predicted Water Consumption:")
    st.write(predicted_consumption[0][0])

    st.subheader("Total Cost (in pesos):")
    st.write(total_cost[0][0])

if __name__ == "__main__":
    main()
