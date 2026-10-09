import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


df = pd.DataFrame({
    "hours": [2, 3, 4, 7, 8, 9],
    "attendance": [55, 60, 65, 85, 90, 95],
    "age": [20, 21, 22, 23, 24, 25],
    "passed": [0, 0, 0, 1, 1, 1]
})

model=DecisionTreeClassifier(random_state=42)
X=df.drop("passed",axis=1)
y=df["passed"]

selector=SelectFromModel(
    estimator=model,  
   threshold="mean"
)
X_selected=selector.fit_transform(X,y)
print("Input columns:", X.columns.tolist())
print("Input shape:", X.shape)
print("Importance scores:", selector.estimator_.feature_importances_)
print("Threshold:", selector.threshold_)
print("Support mask:", selector.get_support())
print("Feature importances:", selector.estimator_.feature_importances_)
print("Selected features:", X.columns[selector.get_support()])