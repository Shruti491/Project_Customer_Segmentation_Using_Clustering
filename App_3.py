# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:12:46 2025

@author: Shruti
"""


import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load scaler and DBSCAN model
scaler = joblib.load("scaler.joblib")
model = joblib.load("IsolationForest_model.joblib")

# Feature list (must match training)
feature_names = [
    'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
    'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
    'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response'
]

# Encoding maps
education_map = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
marital_status_map = {'Single': 0, 'Together': 1, 'Married': 2, 'Divorced': 3, 'Widow': 4}

# Streamlit page setup
st.set_page_config(page_title="Customer Cluster Prediction", layout="wide")
st.title("🔍 Predict Customer Cluster")

with st.sidebar:
    st.header("🧾 Customer Details")
    user_input = {}
    user_input['Year_Birth'] = st.number_input("Year of Birth", 1900, 2025, 1980)
    user_input['Education'] = education_map[st.selectbox("Education Level", list(education_map.keys()), index=0)]
    user_input['Marital_Status'] = marital_status_map[st.selectbox("Marital Status", list(marital_status_map.keys()), index=2)]
    user_input['Income'] = st.number_input("Income", 0, 1_000_000, 55000)
    user_input['Kidhome'] = st.number_input("Kids at Home", 0, 10, 1)
    user_input['Teenhome'] = st.number_input("Teens at Home", 0, 10, 1)
    user_input['Recency'] = st.slider("Days Since Last Purchase", 0, 100, 20)

    st.markdown("---")
    st.subheader("📊 Spending Info")

    default_values = {
        'MntWines': 500,
        'MntFruits': 100,
        'MntMeatProducts': 300,
        'MntFishProducts': 150,
        'MntSweetProducts': 80,
        'MntGoldProds': 250,
        'NumDealsPurchases': 3,
        'NumWebPurchases': 4,
        'NumCatalogPurchases': 2,
        'NumStorePurchases': 5,
        'NumWebVisitsMonth': 4,
        'AcceptedCmp3': 0,
        'AcceptedCmp4': 0,
        'AcceptedCmp5': 0,
        'AcceptedCmp1': 0,
        'AcceptedCmp2': 0,
        'Complain': 0,
        'Response': 1
    }

    for feature in feature_names[7:]:
        default = default_values.get(feature, 0)
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ')}", 0, 2000, default)

# Convert to DataFrame
input_df = pd.DataFrame([[user_input[feature] for feature in feature_names]], columns=feature_names)

# Display input summary
st.subheader("📋 Input Summary")
st.dataframe(input_df)

# Prediction block
if st.button("🚀 Predict Customer Cluster"):
    try:
        scaled_input = scaler.transform(input_df)
        cluster_label = model.fit_predict(scaled_input)

        st.subheader("🎯 Prediction Result")
        if cluster_label[0] == -1:
            st.error("❌ This customer is identified as an outlier (Noise)")
        else:
            st.success(f"✅ This customer belongs to **Cluster {cluster_label[0] + 1}**")



    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

