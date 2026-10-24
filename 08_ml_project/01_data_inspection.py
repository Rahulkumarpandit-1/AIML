from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR /"Telco-Customer-Churn.csv"

df=pd.read_csv(DATA_PATH)

print("dataset_shape:",df.shape)

print("\n First 5 rows:")
print(df.head())

print("\n columns  names:")
print(df.columns.tolist())

print("\n dataset Info")
df.info()

print("\n missing values per column:")
print(df.isnull().sum())        

print("\nduplicate customer id:")
print(df["customerID"].duplicated().sum())  

print("\n churn counts:")
print(df["Churn"].value_counts())


print("\nChurn proportions:")
print(df["Churn"].value_counts(normalize=True).round(3))

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nBlank TotalCharges values:")
print(df["TotalCharges"].astype("string").str.strip().eq("").sum())
blank_mask=df["TotalCharges"].str.strip().eq("")   

print("\n records with blank TotalCharges:")

print(df.loc[
    blank_mask,["customerID","TotalCharges","tenure","MonthlyCharges","Churn"]
])
total_charge_numeric=pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
print("missing value after numeric conversion")
print(total_charge_numeric.isna().sum())    

print("\nTenure values in those records:")
print(df.loc[blank_mask, "tenure"].value_counts())