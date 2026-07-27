import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Read dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Select all categorical columns
categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

# One-Hot Encode all categorical columns
encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(df[categorical_columns])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_columns)
)
print("\n===== Encoded DataFrame =====")
print(encoded_df.head())

# Numerical columns
numerical_df = df[["reading score", "writing score"]]

print("\n===== Numerical Data =====")
print(numerical_df.head())

# Combine encoded and numerical features
X = pd.concat([encoded_df, numerical_df], axis=1)

print("\n===== Final Feature Matrix =====")
print(X.head())

print("\nShape of Feature Matrix:")
print(X.shape)

# One-Hot Encode all categorical columns
encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(df[categorical_columns])
joblib.dump(encoder, "models/encoder.pkl")

print("Encoder saved successfully!")

print("\n===== Encoded Feature Names =====")
print(encoder.get_feature_names_out(categorical_columns))



# Target
y = df["math score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

joblib.dump(model, "models/student_model.pkl")

print("Model saved successfully!")

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 2))
# ============================
# Histogram of Math Scores
# ============================

plt.figure(figsize=(8,5))

plt.hist(df["math score"], bins=10)

plt.title("Distribution of Math Scores")

plt.xlabel("Math Score")

plt.ylabel("Number of Students")

plt.show()
# ============================
# Scatter Plot
# ============================

plt.figure(figsize=(8,5))

plt.scatter(df["reading score"], df["math score"])

plt.title("Reading Score vs Math Score")

plt.xlabel("Reading Score")

plt.ylabel("Math Score")

plt.show()
# ============================
# Correlation Matrix
# ============================

correlation = df[["math score", "reading score", "writing score"]].corr()

print("\n===== Correlation Matrix =====")
print(correlation)

print("\n===== Loading Saved Model =====")

loaded_model = joblib.load("models/student_model.pkl")
loaded_encoder = joblib.load("models/encoder.pkl")

print("Model loaded successfully!")
print("Encoder loaded successfully!")