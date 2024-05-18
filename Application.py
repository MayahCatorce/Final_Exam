import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = tf.keras.models.load_model('water_consumption_lstm_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Define function to make predictions
def predict_consumption(model, previous_consumption):
    try:
        # Pad or repeat the single timestep value to create a sequence of length 10
        input_data = np.repeat(previous_consumption, 10).reshape((1, 10, 1))
        # Make prediction
        predicted_consumption = model.predict(input_data)
        return predicted_consumption[0][0]  # Assuming a scalar output
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Define function to predict 30 days of consumption
def predict_30_days(model, initial_consumption):
    predictions = []
    current_consumption = initial_consumption

    for _ in range(30):
        predicted_consumption = predict_consumption(model, current_consumption)
        if predicted_consumption is None:
            break
        predictions.append(predicted_consumption)
        current_consumption = predicted_consumption  # Use the latest prediction for the next input

    return predictions

# Streamlit app
def main():
    # Custom CSS to style the background and text
    st.markdown("""
        <style>
            body {
                background: url('blue.jpg') no-repeat center center fixed;
                background-size: cover;
                font-family: 'Serif';
            }
            .serif-font {
                font-family: 'Serif';
                font-size: 24px;
                font-weight: bold;
            }
            .stApp {
                background: url('blue.jpg');
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='serif-font'>Final Exam: Deployment in Cloud</div>", unsafe_allow_html=True)
    st.markdown("<div class='serif-font'>Name: Dhafny Buenafe and Mayah Catorce</div>", unsafe_allow_html=True)
    st.markdown("<div class='serif-font'>Section: CPE32S3</div>", unsafe_allow_html=True)
    st.markdown("<div class='serif-font'>Instructor: Engr. Roman Richard</div>", unsafe_allow_html=True)

    st.title('Consumption Prediction App')

    # Load the pre-trained model
    model = load_model()
    if model is None:
        return

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
        if not daily_consumptions:
            st.error("Error generating 30-day predictions.")
            return

        daily_prices = [1.48 * consumption for consumption in daily_consumptions]

        st.write("### 30-Day Predictions")
        df = pd.DataFrame({
            'Day': [f"Day {i+1}" for i in range(30)],
            'Consumption (cubic meters)': daily_consumptions,
            'Price (Php)': daily_prices
        })
        st.table(df)

        # Calculate monthly totals
        total_consumption = np.sum(daily_consumptions)
        total_price = np.sum(daily_prices)

        st.write("### Monthly Prediction")
        st.write(f"Total consumption for the month: {total_consumption:.2f} cubic meters")
        st.write(f"Total price for the month: {total_price:.2f} Php")

if __name__ == "__main__":
    main()
