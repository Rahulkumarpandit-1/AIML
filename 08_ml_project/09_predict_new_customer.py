
from pathlib import Path
import joblib
import pandas as pd

# 1. Locate and load the saved model
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "churn_pipeline.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]

# 2. Create data for one new customer
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 255.0
}

# 3. Convert the customer dictionary into a DataFrame
customer_df = pd.DataFrame([customer])

# 4. Ensure the feature order matches training
customer_df = customer_df[feature_columns]

# 5. Predict churn probability
yes_index = list(model.classes_).index("Yes")
churn_probability = model.predict_proba(customer_df)[0, yes_index]

# 6. Apply the selected threshold
prediction = "Yes" if churn_probability >= threshold else "No"

print("Churn probability:", round(churn_probability, 3))
print("Predicted churn:", prediction)
