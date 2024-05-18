import streamlit as st
import numpy as np

# Streamlit app
def main():
    # set up the Streamlit app
    st.write("Final Exam: Deployment in Cloud")
    st.write("Name: Dhafny Buenafe and Mayah Catorce")
    st.write("Section: CPE32S3")
    st.write("Instructor: Engr. Roman Richard")

    st.title('Consumption Prediction App')

    # User input for previous consumption
    previous_consumption = st.number_input('Enter your previous consumption (cubic meters):', value=0.0)

    # Calculate price and predictions
    if st.button('Calculate Price'):
        # Calculation for a single input
        price = 1.48 * previous_consumption
        st.write(f'Entered consumption: {previous_consumption} cubic meters')
        st.write(f'Estimated price: {price:.2f} Php')

        # Generate 30-day predictions
        np.random.seed(0)  # For reproducibility
        daily_consumptions = np.random.normal(loc=previous_consumption, scale=0.1 * previous_consumption, size=30)
        daily_consumptions = np.maximum(daily_consumptions, 0)  # Ensure no negative consumption values
        daily_prices = 1.48 * daily_consumptions

        st.write("### 30-Day Predictions")
        for day in range(30):
            st.write(f"Day {day + 1}: Consumption = {daily_consumptions[day]:.2f} cubic meters, Price = {daily_prices[day]:.2f} Php")

        # Calculate monthly totals
        total_consumption = np.sum(daily_consumptions)
        total_price = np.sum(daily_prices)

        st.write("### Monthly Prediction")
        st.write(f"Total consumption for the month: {total_consumption:.2f} cubic meters")
        st.write(f"Total price for the month: {total_price:.2f} Php")

if __name__ == "__main__":
    main()
