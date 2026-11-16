
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Locate and load trained model
MODEL_PATH = Path(__file__).resolve().parent / "pipeline.bin"
model = joblib.load(MODEL_PATH)

# Initialize FastAPI
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts applicant credit risk using machine learning.",
    version="1.0"
)


# Define applicant input
class Applicant(BaseModel):
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_MOBIL: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str | None = None
    CNT_FAM_MEMBERS: float


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Random Forest",
        "version": "1.0"
    }


# Prediction endpoint
@app.post("/predict")
def predict(applicant: Applicant):

    # Convert input into DataFrame
    data = pd.DataFrame([applicant.model_dump()])

    # Generate prediction
    prediction = int(model.predict(data)[0])

    probability = float(
        model.predict_proba(data)[0][1]
    )

    # Interpret result
    risk_level = (
        "Higher Risk" if prediction == 1
        else "Lower Risk"
    )

    return {
        "credit_risk": prediction,
        "risk_level": risk_level,
        "risk_probability": round(probability, 4)
    }
