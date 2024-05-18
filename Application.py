import streamlit as st

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

    # Calculate price
    if st.button('Calculate Price'):
        price = 1.48 * previous_consumption  # Calculate price using the given formula
        st.write(f'Entered consumption: {previous_consumption} cubic meters')
        st.write(f'Estimated price: {price:.2f} Php')

if __name__ == "__main__":
    main()
