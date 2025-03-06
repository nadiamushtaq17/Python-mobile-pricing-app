import os
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📱 Mobile Price Prediction")
st.write("Upload your dataset or use default synthetic data for predictions.")

# Define model file path
model_path = r"C:\python Assign\linear_model.pkl"

# Generate synthetic data
np.random.seed(42)
num_samples = 200
df = pd.DataFrame({
    "RAM": np.random.randint(1, 17, num_samples),
    "Battery": np.random.randint(1000, 6001, num_samples),
    "Storage": np.random.randint(16, 513, num_samples),
    "Camera": np.random.randint(5, 109, num_samples),
})

df["Price"] = (
    df["RAM"] * 30 +
    df["Battery"] * 0.5 +
    df["Storage"] * 2 +
    df["Camera"] * 5 +
    np.random.normal(0, 100, num_samples)
)

# Train and save model if not found
if not os.path.exists(model_path):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Price"]), df["Price"], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    st.warning("Trained a new model since the file was missing. Saved as 'linear_model.pkl'.")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# Sidebar for file upload
st.sidebar.header("Upload Custom Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Use uploaded dataset if available
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded successfully!")

# Sidebar sliders for user input
ram = st.sidebar.slider("RAM (GB)", int(df["RAM"].min()), int(df["RAM"].max()), int(df["RAM"].median()))
battery = st.sidebar.slider("Battery (mAh)", int(df["Battery"].min()), int(df["Battery"].max()), int(df["Battery"].median()))
storage = st.sidebar.slider("Storage (GB)", int(df["Storage"].min()), int(df["Storage"].max()), int(df["Storage"].median()))
camera = st.sidebar.slider("Camera (MP)", int(df["Camera"].min()), int(df["Camera"].max()), int(df["Camera"].median()))

st.write(f"**Selected Specs:** RAM: {ram}GB, Battery: {battery}mAh, Storage: {storage}GB, Camera: {camera}MP")

# Make prediction
if st.button("Predict Price"):
    input_data = np.array([[ram, battery, storage, camera]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")

# Scatter plot with regression line
st.subheader("Feature vs Price Scatter Plot")
x_axis = st.selectbox("Select Feature for X-axis", ["RAM", "Battery", "Storage", "Camera"])
fig, ax = plt.subplots()

ax.scatter(df[x_axis], df["Price"], color='blue', alpha=0.5, label="Data Points")
X = df[[x_axis]]
y = df["Price"]
reg_model = LinearRegression().fit(X, y)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = reg_model.predict(X_range)
ax.plot(X_range, y_pred, color='red', linewidth=2, label="Regression Line")

ax.set_xlabel(x_axis)
ax.set_ylabel("Price ($)")
ax.legend()
st.pyplot(fig)
