from sklearn.linear_model import LinearRegression, Ridge

X = [
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
]

y = [35, 40, 50, 55, 65, 70]

# Ordinary linear regression baseline
linear_model = LinearRegression()
linear_model.fit(X, y)

print("Linear Regression coefficients:", linear_model.coef_)

# Compare different regularization strengths
for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X, y)

    prediction = ridge_model.predict([[4, 78]])

    print(f"\nAlpha: {alpha}")
    print("Coefficients:", ridge_model.coef_)
    print("Prediction:", prediction[0])