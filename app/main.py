from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/churn_model.joblib"
app = FastAPI(title="Churn Intelligence API", version="1.0.0")


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120, examples=[8])
    monthly_charges: float = Field(ge=0, examples=[89.5])
    contract: str = "Month-to-month"
    internet_service: str = "Fiber optic"
    tech_support: str = "No"
    payment_method: str = "Electronic check"


def recommendation(probability):
    if probability >= .70:
        return "High priority: offer a retention call and a tailored discount."
    if probability >= .40:
        return "Medium priority: promote a longer contract or technical-support bundle."
    return "Low priority: maintain engagement through regular customer-success outreach."


@app.get("/health")
def health():
    return {"status": "ready" if MODEL_PATH.exists() else "model_not_trained"}


@app.post("/predict")
def predict(customer: Customer):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model missing. Run python src/train.py first.")
    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba(pd.DataFrame([customer.model_dump()]))[0, 1])
    risk = "high" if probability >= .70 else "medium" if probability >= .40 else "low"
    return {"churn_probability": round(probability, 3), "risk_level": risk, "recommendation": recommendation(probability)}
