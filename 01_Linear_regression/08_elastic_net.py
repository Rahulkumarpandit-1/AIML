from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = [
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
]

y = [35, 40, 50, 55, 65, 70]

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "Elastic Net": ElasticNet(alpha=1.0, l1_ratio=0.5),
}

for name, estimator in models.items():
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X, y)

    prediction = model.predict([[4, 78]])

    fitted_estimator = model.steps[-1][1]

    print(f"\n{name}")
    print("Coefficients:", fitted_estimator.coef_)
    print("Prediction:", prediction[0])