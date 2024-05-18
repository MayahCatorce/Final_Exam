import streamlit as st
import pandas as pd
import tensorflow as tf

# Function to load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('water_consumption_lstm_model.h5')
    return model

# Define function to make predictions
def predict_consumption(model, previous_consumption):
    # Ensure input is in the correct format and shape
    input_data = pd.DataFrame({'previous_consumption': [previous_consumption]})
    input_data = input_data.values.reshape((input_data.shape[0], 1, input_data.shape[1]))

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
    previous_consumption = st.number_input('Enter your previous consumption:', value=0.0)

    # Predict consumption
    if st.button('Predict Consumption'):
        # Make predictions
        predicted_consumption = predict_consumption(model, previous_consumption)
        st.write('Predicted consumption:', predicted_consumption)

if __name__ == "__main__":
    main()
