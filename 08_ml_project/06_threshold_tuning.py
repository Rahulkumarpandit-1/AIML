from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
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
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train,X_val,y_train,y_val=train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full   
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


model.fit(X_train_full,y_train_full)


# print('Accuracy:',round(accuracy_score(y_test,y_pred),4))
# print("\n Confusion matrix:")
# print(confusion_matrix(y_test,y_pred,labels=["No","Yes"]))

# print("\n Classification report:")
# print(classification_report(y_test,y_pred,zero_division=0))

yes_index=list(model.classes_).index("Yes")
y_val_prob=model.predict_proba(X_val)[:,yes_index]



for threshold in [0.35, 0.40, 0.45, 0.50]:
    
    y_val_pred=(y_val_prob>=threshold)
    
    precision=precision_score(
            y_val=="Yes",y_val_pred,zero_division=0
    )
    
    recall = recall_score(
    y_val == "Yes", y_val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val == "Yes", y_val_pred, zero_division=0
    )
    

yes_index=list(model.classes_).index("Yes")
y_test_prob=model.predict_proba(X_test)[:,yes_index]

y_test_pred=(y_test_prob>=0.40)
y_test_pred=["Yes" if pred else "No" for pred in y_test_pred]


print("\nFinal Test Results")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=["No", "Yes"]))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, labels=["No", "Yes"]))