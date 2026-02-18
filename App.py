import streamlit as st
import pandas as pd
import pickle

#Getting the data from trained model.
with open("Model/House_Price_Model.pkl", "rb") as f:
    model = pickle.load(f)

#Setting the page web page title.
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

#Setting the page Title and Details.
st.title("🏠 House Price Prediction")
st.write("Enter the house details to predict the price.")

#User inputs.
AREA = st.number_input("Area(sqft)", min_value=300, max_value=10000, value=1200)
BEDROOMS = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
BATHROOMS = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
FLOORS = st.number_input("Floors", min_value=1, max_value=5, value=1)
YEAR_BUILT = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)

LOCATION = st.selectbox(
    "Location",
    ["Downtown", "Urban", "Suburban", "Rural"]
)

CONDITION = st.selectbox(
    "Condition",
    ["Excellent", "Good", "Fair", "Poor"]
)

GARAGE = st.selectbox(
    "Garage",
    ["Yes", "No"]
)

#Prediction of estimated Final Price.
if st.button("Predict Price 💰"):
    input_data = pd.DataFrame({
        "Area": [AREA],
        "Bedrooms": [BEDROOMS],
        "Bathrooms": [BATHROOMS],
        "Floors": [FLOORS],
        "YearBuilt": [YEAR_BUILT],
        "Location": [LOCATION],
        "Condition": [CONDITION],
        "Garage": [GARAGE]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated House Price: ₹ {prediction:,.2f}")
