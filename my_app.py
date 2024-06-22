import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Load the trained model
loaded_model = pickle.load(open('player_rating_predictor.pkl', 'rb'))

# Define a function for predicting player ratings:

def predict_player_rating(input_data):
    if isinstance(loaded_model, RandomForestRegressor):
        input_data = input_data.values.reshape(1, -1)
        prediction = loaded_model.predict(input_data)
        return prediction[0]
    else:
        st.error("Model is not a RandomForestRegressor")


# Create a Streamlit web application
st.title("Player Rating Predictor (highly endorsed by FIFA)")
st.write("Complete the details below to predict a player's overall rating.")
st.warning("USE the SCALE of [0 - 10]")

# Create input fields for user data  with values from 0 to 10
features = ["Player's Potential", "Player's Value", "Player's Wage", "Player's Movement reactions","Plyer's Dribbling", "Players's Movement acceleration", "layer's Attacking Heading Accuracy", "Player's Movement balance", "Player's Movement Sprint Speed", "Player's Passing", "Player's Pace"]
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}:", min_value=0, max_value=10, value=0)


# Create a button to trigger the prediction
if st.button("Predict Rating"):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = predict_player_rating(input_df)
    st.write(f"Predicted Player Rating: {prediction:.2f}")

