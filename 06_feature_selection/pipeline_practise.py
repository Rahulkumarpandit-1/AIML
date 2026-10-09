import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression           
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline   
from sklearn.metrics import mean_absolute_error  

df = pd.DataFrame({
    "size": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "bedrooms": [2, 2, 3, 3, 3, 4, None, 4],
    "price": [40, 50, 60, 75, 90, 100, 110, 125]
})

X=df[["size", "bedrooms"]]
y=df["price"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model=Pipeline(
    [("imputer",SimpleImputer(strategy="mean")),
      ("scaler",StandardScaler()),
      ("regressor",LinearRegression())
])

model.fit(X_train,y_train)
predictions=model.predict(X_test)

mae=mean_absolute_error(y_test,predictions)

print("Actual prices:",y_test.to_list())
print("predicitons:",predictions.round(2))
print("TEst mae:",round(mae,2))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X rows:", len(X))
print("y rows:", len(y))

assert len(X) == len(y), "X and y must have the same number of rows"