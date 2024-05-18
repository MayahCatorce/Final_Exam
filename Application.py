import streamlit as st
import pandas as pd
import tensorflow as tf

# Function to load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('water_consumption_lstm_model.h5')
    return model

# Define function to make predictions
def predict_consumption(model, previous_consumption):
    # Reshape input to match the model's input shape if necessary
    previous_consumption = tf.expand_dims(previous_consumption, axis=0)
    predicted_consumption = model.predict(previous_consumption)
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
