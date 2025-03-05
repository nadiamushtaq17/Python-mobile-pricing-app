from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
print("Current Working Directory:", os.getcwd())

data = pd.read_csv(r"C:\python Assign\mobilePricing.csv")


# Load dataset
data = pd.read_csv("mobilePricing.csv")  # Replace with actual dataset
X = data[['RAM', 'Battery', 'Storage', 'Camera']]  # Independent variables
y = data['Price']  # Dependent variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
print("Mean Squared Error:", mean_squared_error(y_test, predictions))