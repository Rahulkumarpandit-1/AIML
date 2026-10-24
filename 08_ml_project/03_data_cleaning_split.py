from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split        


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"]=pd.to_numeric(
                   df["TotalCharges"].str.strip(),
                   errors="coerce"
)

df.loc[
    df["TotalCharges"].isnull() & (df["tenure"]==0),
    "TotalCharges"
]=0      

# print("\nMissing values after handling:")
# print(df.isnull().sum())

# print("\n TotalCharges data Type:",df["TotalCharges"].dtype)
# print("\n Detected shape:",df.shape)

# print("\n rows with missing totalchagres")
# print(df.loc[df["TotalCharges"].isnull(),
#             ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

x=df.drop(columns=["customerID","Churn"])
y=df["Churn"]

X_train,X_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("X Train shape:",X_train.shape)
print("X Test shape:",X_test.shape)
print("Y Train shape:",y_train.shape)
print("Y Test shape:",y_test.shape)

print("\n training churn distribution")
print(y_train.value_counts(normalize=True).round(3))

print("\n test churn distribution")
print(y_test.value_counts(normalize=True).round(3))             

