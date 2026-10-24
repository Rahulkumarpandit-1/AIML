from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)
df.loc[
    df["TotalCharges"].isnull() & (df["tenure"] == 0),
    "TotalCharges"
] = 0

# Features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features=X_train.select_dtypes(
        include=["float64","int64"]
).columns.tolist()

categorical_features=X_train.select_dtypes(
        include=["object"]
).columns.tolist()  

numeric_transformer=Pipeline(steps=[(
            "imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
]                       
)
categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor=ColumnTransformer(transformers=[
            ("num",numeric_transformer,numeric_features),
            ("cat",categorical_transformer,categorical_features )
]
                              )
model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
])
X_train_processed=preprocessor.fit_transform(X_train)
X_test_processed=preprocessor.transform(X_test)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print('Accuracy:',round(accuracy_score(y_test,y_pred),4))
print("\n Confusion matrix:")
print(confusion_matrix(y_test,y_pred,labels=["No","Yes"]))

print("\n Classification report:")
print(classification_report(y_test,y_pred,zero_division=0))