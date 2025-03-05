import pickle
import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate and Save CSV Data
csv_filename = "mobilePricing.csv"

if not os.path.exists(csv_filename):
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame({
        "RAM": np.random.randint(2, 16, 100),  # 2GB to 16GB
        "Battery": np.random.randint(2000, 6000, 100),  # 2000mAh to 6000mAh
        "Storage": np.random.choice([32, 64, 128, 256], 100),  # Common storage options
        "Camera": np.random.randint(8, 108, 100),  # 8MP to 108MP
        "Price": np.random.randint(100, 1500, 100)  # Price range in dollars
    })
    data.to_csv(csv_filename, index=False)
    print(f" CSV file '{csv_filename}' generated.")

# Step 2: Load Dataset
data = pd.read_csv(csv_filename)
X = data[['RAM', 'Battery', 'Storage', 'Camera']]  # Features
y = data['Price']  # Target

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Step 6: Save Model
model_filename = "linear_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)
print(f"Model saved as '{model_filename}'.")

# Step 7: Streamlit Web App
st.title("📱 Mobile Price Prediction")

ram = st.number_input("Enter RAM (GB):", min_value=2, max_value=16, value=8, step=1)
battery = st.number_input("Enter Battery Capacity (mAh):", min_value=2000, max_value=6000, value=4000, step=500)
storage = st.selectbox("Select Storage (GB):", [32, 64, 128, 256])
camera = st.number_input("Enter Camera Quality (MP):", min_value=8, max_value=108, value=48, step=4)

if st.button("Predict Price"):
    model = pickle.load(open(model_filename, "rb"))  # Load saved model
    prediction = model.predict(np.array([[ram, battery, storage, camera]]))
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
