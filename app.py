from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model files
model = joblib.load("final_crop_recommendation_model.pkl")
scaler = joblib.load("crop_scaler.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")


# -----------------------------
# Functions
# -----------------------------

def recommend_top_crops(N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    input_scaled = scaler.transform(input_data)
    probabilities = model.predict_proba(input_scaled)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_n]

    results = []
    for i in top_indices:
        results.append({
            "crop": label_encoder.inverse_transform([i])[0],
            "score": round(probabilities[i] * 100, 2)
        })

    return results


def soil_advisory(N, P, K, ph):
    advice = []

    advice.append("Nitrogen level is moderate." if 50 <= N <= 100 else "Adjust nitrogen level.")
    advice.append("Phosphorus level is moderate." if 30 <= P <= 80 else "Adjust phosphorus level.")
    advice.append("Potassium level is moderate." if 30 <= K <= 80 else "Adjust potassium level.")
    advice.append("Soil pH is suitable." if 5.5 <= ph <= 7.5 else "Soil pH adjustment needed.")

    return advice


def department_wise_use(crop):
    return [
        {"department": "Farmers", "support": f"Grow {crop} based on soil conditions."},
        {"department": "Consultants", "support": f"Recommend {crop} to clients."},
        {"department": "Fertilizer Companies", "support": f"Plan nutrients for {crop}."},
        {"department": "Government", "support": f"Support farmers growing {crop}."},
        {"department": "Agri-Tech", "support": f"Use {crop} in advisory systems."}
    ]


def sustainability_output(crop):
    return [
        f"{crop} supports better land use",
        "Reduces fertilizer waste",
        "Improves soil health",
        "Encourages sustainable farming"
    ]


# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        crops = recommend_top_crops(N, P, K, temperature, humidity, ph, rainfall)
        top_crop = crops[0]["crop"]

        result = {
            "crops": crops,
            "soil": soil_advisory(N, P, K, ph),
            "department": department_wise_use(top_crop),
            "sustainability": sustainability_output(top_crop)
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)