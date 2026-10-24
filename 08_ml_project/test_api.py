
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

VALID_CUSTOMER = {
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
    "TotalCharges": 255.0,
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_valid_prediction():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["predicted_churn"] in ["Yes", "No"]
    assert result["threshold"] == 0.4


def test_negative_tenure_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["tenure"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_invalid_senior_citizen_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["SeniorCitizen"] = 2

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

def test_invalid_contract_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["Contract"] = "Random"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
