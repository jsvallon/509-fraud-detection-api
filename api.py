from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("fraud_model.pkl")  # Ensure `model.pkl` is in the correct path

# Define the request data model
class TransactionData(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float
    feature_22: float
    feature_23: float
    feature_24: float
    feature_25: float
    feature_26: float
    feature_27: float
    feature_28: float
    feature_29: float
    feature_30: float

# Prediction endpoint
@app.post("/predict")
async def predict(transaction: TransactionData):
    try:
        # Convert input to numpy array
        data = np.array([[
            transaction.amount,
            transaction.oldbalanceOrg,
            transaction.newbalanceOrig,
            transaction.oldbalanceDest,
            transaction.newbalanceDest,
            transaction.feature_6,
            transaction.feature_7,
            transaction.feature_8,
            transaction.feature_9,
            transaction.feature_10,
            transaction.feature_11,
            transaction.feature_12,
            transaction.feature_13,
            transaction.feature_14,
            transaction.feature_15,
            transaction.feature_16,
            transaction.feature_17,
            transaction.feature_18,
            transaction.feature_19,
            transaction.feature_20,
            transaction.feature_21,
            transaction.feature_22,
            transaction.feature_23,
            transaction.feature_24,
            transaction.feature_25,
            transaction.feature_26,
            transaction.feature_27,
            transaction.feature_28,
            transaction.feature_29,
            transaction.feature_30
        ]])

        # Make a prediction
        fraud_probability = model.predict_proba(data)[0][1]  # Assuming model supports predict_proba
        fraudulent = fraud_probability > 0.5  # Example threshold

        return {
            "fraud_probability": round(fraud_probability, 2),
            "fraudulent": fraudulent
        }
    
    except Exception as e:
        return {"detail": f"Error processing prediction: {str(e)}"}

# Health check route
@app.get("/")
async def root():
    return {"message": "FastAPI Model is Running!"}
