# Customer Churn Prediction

## Overview
An end-to-end machine learning project that predicts whether a telecom customer is likely to leave the service. It includes data analysis, preprocessing, model training, threshold tuning, error analysis, model saving, and a FastAPI prediction service.

## Dataset
- **Source:** IBM Telco Customer Churn dataset
- **Rows:** 7,043 customers
- **Target:** `Churn` (`Yes` or `No`)
- `customerID` is excluded from model features.
- Missing `TotalCharges` values are handled during data cleaning.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- FastAPI and Uvicorn
- Pytest and HTTPX-compatible test client

## Project Workflow
1. `01_data_inspection.py` — inspect the dataset.
2. `02_eda.py` — explore distributions and relationships.
3. `03_data_cleaning_split.py` — clean data and create splits.
4. `04_preprocessing.py` — prepare numeric and categorical features.
5. `05_logistic_regression.py` — train the baseline model.
6. `06_threshold_tuning.py` — select a classification threshold.
7. `07_error_analysis.py` — examine prediction errors.
8. `08_save_model.py` — save the fitted pipeline.
9. `09_predict_new_customer.py` — predict churn for a new customer.
10. `api.py` — serve predictions through FastAPI.
11. `test_api.py` — test API behavior and input validation.

## Model and Preprocessing
The model uses Logistic Regression in a Scikit-learn pipeline.

- Numeric features: median imputation and standard scaling.
- Categorical features: most-frequent imputation and one-hot encoding.
- Unknown categorical values are handled by the encoder.
- The selected decision threshold is **0.40**, chosen using validation data.

## Test Results
Final test-set results at threshold 0.40:

| Metric | Result |
|---|---:|
| Accuracy | 77.71% |
| Churn precision | ~0.57 |
| Churn recall | ~0.67 |
| Churn F1-score | ~0.61 |

Confusion matrix, with labels `No` then `Yes`:

```text
[[845, 190],
 [124, 250]]
```

The model identified 250 actual churners and missed 124. It also flagged 190 customers who did not churn. Predictions are estimates, not guarantees.

## Installation

Run commands from this project directory.

```bash
python -m pip install -r requirements.txt
```

## Run a Prediction Script

```bash
python 09_predict_new_customer.py
```

## Run the API

```bash
python -m uvicorn api:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

- `GET /` — check whether the API is running.
- `POST /predict` — submit customer details and receive a churn probability and prediction.

Stop the local server with `Ctrl+C`.

## Run Automated Tests

```bash
python -m pytest -v
```

The current API test suite contains five tests covering the home endpoint, valid predictions, and rejection of invalid inputs.

## Saved Model
The fitted pipeline and prediction threshold are saved in `churn_pipeline.joblib`. Keep this artifact available when running the prediction script or API.

## Limitations and Future Improvements
- Performance depends on the available dataset and may not generalize to every telecom provider.
- The classification threshold affects the balance between catching churners and generating false alarms.
- Future work could include stronger model comparisons, monitoring, deployment, and testing on new data.
