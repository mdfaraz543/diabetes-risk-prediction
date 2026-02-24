import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("diabetes.csv", header=None)

df.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# 🔥 Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "best_model.joblib")

print("Model retrained successfully!")