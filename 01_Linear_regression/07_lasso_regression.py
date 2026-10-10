from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
    "Linear Regression": LinearRegression(),
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Lasso": make_pipeline(
        StandardScaler(),
        Lasso(alpha=1.0)
    ),
}

for name, model in models.items():
    model.fit(X, y)

    prediction = model.predict([[4, 78]])

    print(f"\n{name}")
    print("Prediction:", prediction[0])

    if name == "Linear Regression":
        print("Coefficients:", model.coef_)
    else:
        print("Coefficients:", model.named_steps[
            "ridge" if name == "Ridge" else "lasso"
        ].coef_)