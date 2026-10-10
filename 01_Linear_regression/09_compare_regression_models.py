from sklearn.model_selection import  train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

X = [
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
    [7, 82],
    [8, 90],
    [9, 88],
    [10, 95],
]

y = [35, 40, 50, 55, 65, 70, 72, 80, 85, 92]


X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "Elastic Net": ElasticNet(alpha=1.0, l1_ratio=0.5),
}


for name, estimator in models.items():
    if name == "Linear Regression":
        model = estimator
    else:
        model = make_pipeline(StandardScaler(), estimator)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("Predictions:", predictions)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)