# train_model.py

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths
DATA_PATH = "data/fitness_dataset.csv"
MODEL_PATH = "model/fitness_pipeline.pkl"
CM_PATH = "static/confusion_matrix.png"

os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Load dataset
df = pd.read_csv(r"/Users/aadisreekumar/Desktop/project/data/fitness_dataset (1).csv")

# Cleaning
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
df["gender"] = df["gender"].astype(str).str.upper().str[0]
df["smokes"] = df["smokes"].astype(str).str.lower().apply(
    lambda x: "yes" if x in ["yes", "1", "y"] else "no"
)

FEATURES = [
    "age", "height_cm", "weight_kg", "heart_rate",
    "blood_pressure", "sleep_hours",
    "nutrition_quality", "activity_index",
    "smokes", "gender"
]

TARGET = "is_fit"

X = df[FEATURES]
y = df[TARGET]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing
cat_features = ["smokes", "gender"]
num_features = [col for col in FEATURES if col not in cat_features]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", StandardScaler(), num_features)
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["Not Fit", "Fit"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Fitness Classification")
plt.savefig(CM_PATH)
plt.show()

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved:", MODEL_PATH)
print("Confusion matrix saved at:", CM_PATH)