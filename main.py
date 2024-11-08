import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.markdown(
    """
    <style>
    .centered-header {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    </style>
    <div class="centered-header">
        <h1>Could you be suffering from PCOS?</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader('Fill in the following details.')

col1, col2 = st.columns(2)
user_name = col1.text_input('Your Name', help="Please enter your full name.")
age = col2.number_input('Age (in Years)', min_value=10, max_value=100, help="Please enter your age. Age must be between 10 and 100.")  # Part of model input for prediction

# Numeric inputs for Weight and Height
weight = st.number_input('Weight (in Kg)', min_value=10, max_value=200, help="Please enter your weight in kilograms.")
height = st.number_input('Height (in Cm)', min_value=50, max_value=250, help="Please enter your height.")

# Select box for period cycle frequency
period_cycle = st.selectbox('After how many months do you get your periods?', ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], help="Select 1 for every month/regular period cycle.")

# Yes/No questions
recent_weight_gain = st.selectbox('Have you gained weight recently?', ['Yes', 'No'], help="Have you gained weight recently?")
excessive_hair_growth = st.selectbox('Do you have excessive body/facial hair growth ?', ['Yes', 'No'], help="Do you have excessive body or facial hair growth?")
skin_darkening = st.selectbox('Are you noticing skin darkening recently?', ['Yes', 'No'], help="Are you noticing skin darkening recently?")
hair_loss = st.selectbox('Do you have hair loss/hair thinning/baldness ?', ['Yes', 'No'], help="Do you have hair loss, thinning, or baldness?")
acne = st.selectbox('Do you have pimples/acne on your face/jawline ?', ['Yes', 'No'], help="Do you have pimples or acne on your face/jawline?")
fast_food = st.selectbox('Do you eat fast food regularly ?', ['Yes', 'No'], help="Do you eat fast food regularly?")
exercise = st.selectbox('Do you exercise on a regular basis ?', ['Yes', 'No'], help="Do you exercise regularly?")
mood_swings = st.selectbox('Do you experience mood swings ?', ['Yes', 'No'], help="Do you experience mood swings?")
regular_periods = st.selectbox('Are your periods regular ?', ['Yes', 'No'], help="Are your periods regular?")

# Numeric input for period duration
period_duration = st.number_input('How long does your period last? (in Days)', min_value=1, max_value=10, help="Please enter the duration of your period in days.")

# Creating a list for model input
input_variables = [
    age, weight, height, period_cycle, recent_weight_gain, excessive_hair_growth, 
    skin_darkening, hair_loss, acne, fast_food, exercise, mood_swings, 
    regular_periods, period_duration
]

model_input_dict = {'Yes': 1, 'No': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12}
model_list = [model_input_dict[str(y)] if str(y) in model_input_dict else y for y in input_variables]

# Importing model and preparing dataframe for it
model = joblib.load('final_model_2.joblib')  # Assuming this is the correct model
# Create a list of your desired column names
columns = [
    'Age (in Years)', 'Weight (in Kg)', 'Height (in Cm / Feet)', 
    'After how many months do you get your periods?\n(select 1- if every month/regular)', 
    'Have you gained weight recently?', 
    'Do you have excessive body/facial hair growth ?', 
    'Are you noticing skin darkening recently?', 
    'Do have hair loss/hair thinning/baldness ?', 
    'Do you have pimples/acne on your face/jawline ?',
    'Do you eat fast food regularly ?',
    'Do you exercise on a regular basis ?',
    'Do you experience mood swings ?',
    'Are your periods regular ?',
    'How long does your period last ? (in Days)\nexample- 1,2,3,4.....'
]
df = pd.DataFrame(np.array([model_list]), columns=columns)

# Initialize session state for prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def predict():
    prediction = model.predict(df)[0]
    st.session_state.prediction = prediction

st.button(label='Predict', on_click=predict, help="Click to predict your PCOS risk based on the provided information.")

# Display the prediction result
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.header(f'{user_name}, you may have PCOS.')
        st.subheader("It is recommended that you consult with a healthcare provider for further evaluation and diagnosis.")
    else:
        st.header(f'{user_name}, you are unlikely to have PCOS.')
        st.subheader("However, if you experience symptoms, it is still a good idea to consult a healthcare provider for confirmation.")

# Disclaimer at the end of the page
st.markdown(
    """
    <hr>
    <p style="font-size: 12px; text-align: center; color: gray;">
        Disclaimer: This prediction is based on the information provided and does not substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your doctor or other qualified health provider with any questions you may have regarding a medical condition.
    </p>
    """,
    unsafe_allow_html=True
)

