# ============================================================
# Customer Churn Prediction - Streamlit Web App
# ============================================================


# ------------------------------------------------------------
# 1. Import required libraries
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle


# ------------------------------------------------------------
# 2. Load the trained ANN model
# ------------------------------------------------------------

# Load the ANN model that we trained earlier
model = tf.keras.models.load_model("model.h5")


# ------------------------------------------------------------
# 3. Load the saved preprocessing objects
# ------------------------------------------------------------

# Load the LabelEncoder used for Gender
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)


# Load the OneHotEncoder used for Geography
with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)


# Load the StandardScaler used during model training
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ------------------------------------------------------------
# 4. Create the Streamlit application title
# ------------------------------------------------------------

st.title("Customer Churn Prediction")


# ------------------------------------------------------------
# 5. Create input fields for the user
# ------------------------------------------------------------

# Geography dropdown
geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)


# Gender dropdown
gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)


# Age slider
age = st.slider(
    "Age",
    18,
    100,
    30
)


# Balance input
balance = st.number_input(
    "Balance",
    min_value=0.0
)


# Credit Score input
credit_score = st.number_input(
    "Credit Score",
    min_value=0
)


# Estimated Salary input
estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0
)


# Tenure slider
tenure = st.slider(
    "Tenure",
    0,
    10,
    5
)


# Number of Products slider
num_of_products = st.slider(
    "Number of Products",
    1,
    4,
    1
)


# Has Credit Card
# 0 = No, 1 = Yes
has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)


# Is Active Member
# 0 = No, 1 = Yes
is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# ------------------------------------------------------------
# 6. Prepare the user input
# ------------------------------------------------------------

# Store all user inputs inside a dictionary
input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}


# Convert the dictionary into a Pandas DataFrame
# The model's preprocessing expects tabular data
input_df = pd.DataFrame([input_data])


# ------------------------------------------------------------
# 7. Encode Gender
# ------------------------------------------------------------

# Convert Gender from text to numerical value
# Example:
# Male/Female → 0/1
input_df["Gender"] = label_encoder_gender.transform(
    input_df["Gender"]
)


# ------------------------------------------------------------
# 8. One-hot encode Geography
# ------------------------------------------------------------

# Transform Geography using the same OneHotEncoder
# that was fitted during training
geo_encoded = onehot_encoder_geo.transform(
    input_df[["Geography"]]
).toarray()


# Convert the encoded Geography array into a DataFrame
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ["Geography"]
    )
)


# ------------------------------------------------------------
# 9. Combine the encoded data
# ------------------------------------------------------------

# Remove the original Geography column
# because we now have its one-hot encoded columns
#
# Then combine the remaining features with
# the encoded Geography columns
input_df = pd.concat(
    [
        input_df.drop("Geography", axis=1),
        geo_encoded_df
    ],
    axis=1
)


# ------------------------------------------------------------
# 10. Scale the input data
# ------------------------------------------------------------

# Use the SAME scaler that was fitted on training data
#
# IMPORTANT:
# We use transform() and NOT fit_transform()
# because we don't want to fit the scaler again.
input_scaled = scaler.transform(input_df)


# ------------------------------------------------------------
# 11. Make prediction using the ANN model
# ------------------------------------------------------------

# Pass the processed customer data to the ANN
prediction = model.predict(input_scaled)


# ------------------------------------------------------------
# 12. Extract churn probability
# ------------------------------------------------------------

# The sigmoid output gives the probability of churn
prediction_probability = prediction[0][0]


# Display the probability in the Streamlit app
st.write(
    "Churn Probability:",
    prediction_probability
)


# ------------------------------------------------------------
# 13. Convert probability into final prediction
# ------------------------------------------------------------

# If probability is greater than 0.5,
# classify the customer as likely to churn.
#
# Otherwise, classify the customer as
# not likely to churn.

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")