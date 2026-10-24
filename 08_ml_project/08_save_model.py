
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. Set project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_DIR / "churn_pipeline.joblib"

# 2. Load and clean data
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

df.loc[
    df["TotalCharges"].isna() & (df["tenure"] == 0),
    "TotalCharges"
] = 0

# 3. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# 4. Recreate the same training and test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Identify feature types
numeric_features = X_train_full.select_dtypes(
    include=["float64", "int64"]
).columns.tolist()

categorical_features = X_train_full.select_dtypes(
    include=["object"]
).columns.tolist()

# 6. Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Build and train the complete pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train_full, y_train_full)

# 8. Save the fitted pipeline and prediction threshold
artifact = {
    "model": model,
    "threshold": 0.40,
    "feature_columns": X.columns.tolist()
}

joblib.dump(artifact, MODEL_PATH)

print("Model saved successfully.")
print("Saved to:", MODEL_PATH)
print("Threshold:", artifact["threshold"])
