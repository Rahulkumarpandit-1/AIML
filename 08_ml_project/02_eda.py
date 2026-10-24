
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],
                                   errors="coerce") 
# # churn ditribution
# print("\nchurn distribution:")
# print(df["Churn"].value_counts())       

# df["Churn"].value_counts().plot(kind="bar",title="Churn Distribution")
                              
# plt.xlabel("Churn")
# plt.ylabel("No of customers")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

# contract_churn=pd.crosstab(df["Contract"],
#                            df["Churn"],
#                            normalize="index")*100

# print("\n churn percentage by contract type:")
# print(contract_churn.round(2))

# contract_churn.plot(kind="bar",stacked=True,
#                     title="Churn Percentage by Contract Type") 

# plt.xlabel("Contract Type")
# plt.ylabel("Percentage of Customers")
# plt.xticks(rotation=0)
# plt.legend(title="Churn")
# plt.tight_layout()
# plt.show()
   
#  Tenure by churn status
# print("\nTenure statistics by churn:")
# print(df.groupby("Churn")["tenure"].describe().round(2))

# df.boxplot(column="tenure", by="Churn")
# plt.title("Customer Tenure by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn")
# plt.ylabel("Tenure (months)")
# plt.tight_layout()
# plt.show()    

# print("\nMonthly charges by churn:")
# print(
#     df.groupby("Churn")["MonthlyCharges"]
#     .mean()
#     .round(2)
# )

# df.boxplot(column="MonthlyCharges", by="Churn")
# plt.title("Monthly Charges by Churn Status")
# plt.suptitle("")
# plt.xlabel("Churn")
# plt.ylabel("Monthly Charges")
# plt.tight_layout()
# plt.show()

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

# tenure_churn = pd.crosstab(
#     df["tenure_group"],
#     df["Churn"],
#     normalize="index"
# ) * 100

# print("\nChurn percentage by tenure group:")
# print(tenure_churn.round(2))

# tenure_churn["Yes"].plot(
#     kind="bar",
#     title="Churn Rate by Tenure Group"
# )
# plt.xlabel("Tenure (months)")
# plt.ylabel("Churn Rate (%)")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

# Churn rate by tenure group
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

tenure_churn = pd.crosstab(
    df["tenure_group"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by tenure group:")
print(tenure_churn.round(2))

tenure_churn["Yes"].plot(
    kind="bar",
    title="Churn Rate by Tenure Group"
)
plt.xlabel("Tenure (months)")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()