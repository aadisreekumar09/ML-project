# app.py
from flask import Flask, render_template, request, url_for, redirect
import os
import pickle
import pandas as pd

app = Flask(__name__)

# --- Load model ---
MODEL_PATH = os.path.join("model", "fitness_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# --- Helpers ---
def safe_float(val, name):
    """Convert form value to float and raise readable error if not possible."""
    if val is None:
        raise ValueError(f"Field '{name}' is missing.")
    s = str(val).strip()
    if s == "":
        raise ValueError(f"Field '{name}' is required.")
    try:
        return float(s)
    except Exception:
        raise ValueError(f"Field '{name}' must be a number. Received: {val!r}")


# --- Routes ---
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_text = None
    prediction_status = None
    error_text = None

    if request.method == "POST":
        try:
            age = safe_float(request.form.get("age"), "age")
            height_cm = safe_float(request.form.get("height_cm"), "height_cm")
            weight_kg = safe_float(request.form.get("weight_kg"), "weight_kg")
            heart_rate = safe_float(request.form.get("heart_rate"), "heart_rate")
            blood_pressure = safe_float(request.form.get("blood_pressure"), "blood_pressure")
            sleep_hours = safe_float(request.form.get("sleep_hours"), "sleep_hours")
            nutrition_quality = safe_float(request.form.get("nutrition_quality"), "nutrition_quality")
            activity_index = safe_float(request.form.get("activity_index"), "activity_index")

            gender = request.form.get("gender", "").strip()
            smokes = request.form.get("smokes", "").strip()

            if gender == "":
                raise ValueError("Gender is required.")
            if smokes == "":
                raise ValueError("Smokes is required.")

            input_df = pd.DataFrame({
                "age": [age],
                "height_cm": [height_cm],
                "weight_kg": [weight_kg],
                "heart_rate": [heart_rate],
                "blood_pressure": [blood_pressure],
                "sleep_hours": [sleep_hours],
                "nutrition_quality": [nutrition_quality],
                "activity_index": [activity_index],
                "smokes": [smokes],
                "gender": [gender],
            })

            pred = int(model.predict(input_df)[0])
            prob = None
            try:
                prob = float(model.predict_proba(input_df)[0][1])
            except Exception:
                prob = None

            if pred == 1:
                prediction_status = "fit"
                prediction_text = "Predicted: FIT"
                if prob is not None:
                    prediction_text += f" (confidence: {prob*100:.1f}%)"
            else:
                prediction_status = "not_fit"
                prediction_text = "Predicted: NOT FIT"
                if prob is not None:
                    prediction_text += f" (fitness probability: {prob*100:.1f}%)"

        except Exception as e:
            error_text = str(e)

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        prediction_status=prediction_status,
        error_text=error_text
    )


if __name__ == "__main__":
    app.run(debug=True)