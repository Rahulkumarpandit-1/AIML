from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer       
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer    
from sklearn.pipeline import Pipeline       
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"]=pd.to_numeric(
    df["TotalCharges"].str.strip()
    ,errors="coerce"
)


df.loc[df["TotalCharges"].isnull() & (df["tenure"]==0),
    "TotalCharges"
]=0

X=df.drop(columns=["customerID","Churn"])
y=df["Churn"]   

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
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

X_train_processed=preprocessor.fit_transform(X_train)
X_test_processed=preprocessor.transform(X_test)

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)