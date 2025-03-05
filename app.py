import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model_path = r"C:\python Assign\linear_model.pkl"

if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    st.error("Model file not found. Please check the file path.")
    st.stop()

st.title("Mobile Price Prediction")

# Sidebar for input sliders
st.sidebar.header("Set Parameters")
ram = st.sidebar.slider("RAM (GB)", 1, 16, 4)
battery = st.sidebar.slider("Battery Capacity (mAh)", 1000, 6000, 3000)
storage = st.sidebar.slider("Storage (GB)", 16, 512, 64)
camera = st.sidebar.slider("Camera Quality (MP)", 5, 108, 12)
price = st.sidebar.slider("Price ($)", 100, 1000, 500)

# Display selected values
st.write(f"**Selected Values:** RAM: {ram}GB, Battery: {battery}mAh, Storage: {storage}GB, Camera: {camera}MP, Price: ${price}")

if st.button("Predict"):
    prediction = model.predict(np.array([[ram, battery, storage, camera]]))
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")

# Dynamic scatter plot
st.subheader("Scatter Plot of Features vs Price")
x_axis = st.selectbox("Select X-axis Feature", ["ram", "battery", "storage", "camera"])
y_axis = "price"

data = pd.DataFrame({
    "ram": [ram],
    "battery": [battery],
    "storage": [storage],
    "camera": [camera],
    "price": [price]
})

if x_axis in data.columns and y_axis in data.columns:
    fig, ax = plt.subplots()
    ax.scatter(data[x_axis], data[y_axis], color='blue', alpha=0.5, label='Data Points')
    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Price")
    ax.set_title(f"{x_axis.capitalize()} vs Price")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("Selected feature is not available in the dataset.")
