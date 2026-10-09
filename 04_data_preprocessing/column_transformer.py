from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler  
from sklearn.pipeline import Pipeline   
from sklearn.linear_model import LogisticRegression
import pandas as pd

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "salary"]),
        ("cat", OneHotEncoder(), ["city"])
    ]
)
model_pipeline = Pipeline([
    ("preprocessing",preprocessor),
    ("model",LogisticRegression()   )
])
df = pd.DataFrame({
    "age": [20, 25, 30, 35],
    "salary": [20000, 40000, 60000, 80000],
    "city": ["Delhi", "Mumbai", "Guwahati", "Delhi"]
})
y=[0,1,0,1]

X = df[["age", "salary", "city"]]
model_pipeline.fit(X,y)

predictions=model_pipeline.predict(X)   
print(predictions)