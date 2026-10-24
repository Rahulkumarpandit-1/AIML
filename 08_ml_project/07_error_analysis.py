
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load and clean data
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

df.loc[
    df["TotalCharges"].isna() & (df["tenure"] == 0),
    "TotalCharges"
] = 0

# 2. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# 3. Recreate the same train/validation/test splits
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

# 4. Identify feature types using training data only
numeric_features = X_train.select_dtypes(
    include=["float64", "int64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# 5. Build preprocessing pipelines
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

# 6. Build the complete model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Fit on training subset, then choose threshold on validation data
model.fit(X_train, y_train)

yes_index = list(model.classes_).index("Yes")
y_val_prob = model.predict_proba(X_val)[:, yes_index]

threshold = 0.40

# 8. Refit on the full training pool
model.fit(X_train_full, y_train_full)

# 9. Predict on the untouched test set
y_test_prob = model.predict_proba(X_test)[:, yes_index]
y_test_pred = ["Yes" if p >= threshold else "No"
               for p in y_test_prob]

# 10. Build an error-analysis table
results = X_test.copy()
results["customerID"] = df.loc[X_test.index, "customerID"]
results["Actual"] = y_test
results["Predicted"] = y_test_pred
results["ChurnProbability"] = y_test_prob

false_positives = results[
    (results["Actual"] == "No") &
    (results["Predicted"] == "Yes")
]

false_negatives = results[
    (results["Actual"] == "Yes") &
    (results["Predicted"] == "No")
]
print("\nActual churn rate by contract:")
print(
    results.groupby("Contract")["Actual"]
    .apply(lambda s: (s == "Yes").mean())
    .mul(100)
    .round(2)
)

print("\nModel recall by contract:")
for contract, group in results.groupby("Contract"):
    actual_churners = group["Actual"] == "Yes"
    if actual_churners.sum() > 0:
        recall = (
            (group["Predicted"] == "Yes") & actual_churners
        ).sum() / actual_churners.sum()

        print(f"{contract}: {recall:.3f}")
        
        
        
print("\n contract wise prediction count")
print(pd.crosstab(
    results["Contract"],
    [results["Actual"], results["Predicted"]],
    rownames=["Contract"],
    colnames=["Actual","Predicted"],
    margins=True
))  

print("\nChurn probability by contract and actual outcome:")

print(
    results.groupby(["Contract","Actual"])["ChurnProbability"]
    .agg(["count","mean","min","max"])
    .round(3)
)  
# print("\nFinal Test Confusion Matrix:")
# print(confusion_matrix(y_test, y_test_pred, labels=["No", "Yes"]))

# print("\nFinal Test Classification Report:")
# print(classification_report(
#     y_test, y_test_pred, labels=["No", "Yes"], zero_division=0
# ))

# print("\nError Counts")
# print("False positives:", len(false_positives))
# print("False negatives:", len(false_negatives))

# # 11. Compare the characteristics of both error groups
# numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

# print("\nAverage values among error groups:")
# print("\nFalse positives:")
# print(false_positives[numeric_columns].mean().round(2))

# print("\nFalse negatives:")
# print(false_negatives[numeric_columns].mean().round(2))

# print("\nFalse negatives by contract:")
# print(false_negatives["Contract"].value_counts())

# print("\nFalse positives by contract:")
# print(false_positives["Contract"].value_counts())

# # 12. Show a few missed churners for inspection
# print("\nFirst 10 missed churners:")
# print(
#     false_negatives[
#         ["customerID", "Contract", "tenure",
#          "MonthlyCharges", "ChurnProbability"]
#     ].head(10).to_string(index=False)
# )
