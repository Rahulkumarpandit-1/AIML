from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from pydantic import BaseModel, Field



PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "churn_pipeline.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]

app=FastAPI(title="Customer Churn Prediction API")



class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

@app.get("/")
def Home():
    return {"message":"Customer churn prediction pipeline is running"}    

@app.post("/predict")
def predict_churn(customer: Customer):
    customer_data=customer.model_dump()
    
    customer_df = pd.DataFrame([customer_data])
    customer_df = customer_df[feature_columns]
    
    yes_index=list(model.classes_).index("Yes")
    churn_probability=model.predict_proba(customer_df)[0,yes_index]
    
    prediction=(
        "Yes" if churn_probability>=threshold else "No"
    )
    
    return {
        "churn_probability":round(float(churn_probability),3),
        "predicted_churn":prediction,
        "threshold":threshold
    }