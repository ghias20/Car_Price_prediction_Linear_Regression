import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load saved model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Predictor")
st.markdown("Predict car prices based on **Horsepower** using Linear Regression.")

# --- Sidebar Inputs ---
st.header("Input Features")
horsepower = st.slider("Horsepower", min_value=50, max_value=300, value=110, step=5)

# --- Prediction ---
if st.button("Predict Price"):
    input_data = np.array([[horsepower]])
    input_scaled = scaler.transform(input_data)      # use the SAME fitted scaler
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"💰 Predicted Price for {horsepower} HP: **${predicted_price:,.2f}**")
    # --- Visualization ---
    hp_range = np.linspace(50, 300, 200).reshape(-1, 1)
    hp_scaled = scaler.transform(hp_range)
    price_range = model.predict(hp_scaled)
    fig, ax = plt.subplots()
    ax.plot(hp_range, price_range, color='steelblue', label='Regression Line')
    ax.scatter([horsepower], [predicted_price], color='red', zorder=5, label=f'Your Input ({horsepower} HP)')
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Horsepower vs Predicted Car Price")
    ax.legend()
    st.pyplot(fig)
st.markdown("---")
st.caption("Model: Linear Regression | Feature: Horsepower | Scaled with StandardScaler")
