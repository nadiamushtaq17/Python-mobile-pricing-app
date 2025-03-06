import os
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📱 Mobile Price Prediction")
st.write("Upload your dataset or use default synthetic data for predictions.")

# Load trained model
model_path = r"C:\python Assign\linear_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
else:
    st.error("Model file not found. Please check the file path.")
    st.stop()

# Sidebar for file upload
st.sidebar.header("Upload Custom Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Load dataset from file or generate synthetic data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # ✅ Load dataset
    st.sidebar.success("Dataset loaded successfully!")
else:
    st.sidebar.warning("No dataset uploaded. Using synthetic data.")
    
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 200
    df = pd.DataFrame({
        "RAM": np.random.randint(1, 17, num_samples),
        "Battery": np.random.randint(1000, 6001, num_samples),
        "Storage": np.random.randint(16, 513, num_samples),
        "Camera": np.random.randint(5, 109, num_samples),
    })

    # Simulated price generation (Linear relationship with some noise)
    df["Price"] = (
        df["RAM"] * 30 +
        df["Battery"] * 0.5 +
        df["Storage"] * 2 +
        df["Camera"] * 5 +
        np.random.normal(0, 100, num_samples)  # Adding some random noise
    )

# Sidebar sliders for input values (using dataset values)
ram = st.sidebar.slider("RAM (GB)", int(df["RAM"].min()), int(df["RAM"].max()), int(df["RAM"].median()))
battery = st.sidebar.slider("Battery Capacity (mAh)", int(df["Battery"].min()), int(df["Battery"].max()), int(df["Battery"].median()))
storage = st.sidebar.slider("Storage (GB)", int(df["Storage"].min()), int(df["Storage"].max()), int(df["Storage"].median()))
camera = st.sidebar.slider("Camera Quality (MP)", int(df["Camera"].min()), int(df["Camera"].max()), int(df["Camera"].median()))

# Display selected values
st.write(f"**Selected Values:** RAM: {ram}GB, Battery: {battery}mAh, Storage: {storage}GB, Camera: {camera}MP")

# Price Prediction
if st.button("Predict"):
    input_data = np.array([[ram, battery, storage, camera]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")

# Scatter plot with regression line
st.subheader("Scatter Plot of Features vs Price")
x_axis = st.selectbox("Select X-axis Feature", ["RAM", "Battery", "Storage", "Camera"])
y_axis = "Price"

if x_axis in df.columns and y_axis in df.columns:
    fig, ax = plt.subplots()
    
    # Scatter plot
    ax.scatter(df[x_axis], df[y_axis], color='blue', alpha=0.5, label='Data Points')
    
    # Regression line
    X = df[[x_axis]]
    y = df[y_axis]
    reg_model = LinearRegression().fit(X, y)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = reg_model.predict(X_range)
    ax.plot(X_range, y_pred, color='red', linewidth=2, label='Regression Line')

    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{x_axis.capitalize()} vs Price")
    ax.legend()
    
    st.pyplot(fig)
else:
    st.error("Selected feature is not available in the dataset.")
