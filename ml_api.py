from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

print("Loading trained ML model...")


model = joblib.load("fraud_model.pkl")

print("Model loaded successfully")


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection AI Agent Running"
    }

@app.post("/predict")
def predict(data: dict):

    try:
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "FRAUD" if prediction == 1 else "SAFE"

        return {
            "result": result,
            "fraud": int(prediction),
            "confidence": float(probability)
        }

    except Exception as e:
        return {
            "error": str(e)
        }