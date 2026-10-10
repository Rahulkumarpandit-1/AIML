import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([15, 25, 40, 60, 85, 115])

# PolynomialFeatures(degree=2) transforms each input \(x\) into [x, x²]
# LinearRegression() learns a coefficient for each transformed feature.
# make_pipeline() connects these steps so the transformation happens automatically during training and prediction.


model=make_pipeline(
    PolynomialFeatures(degree=2,include_bias=False),
    LinearRegression()
)

model.fit(X,y)

prediction=model.predict([[7]])

print("Prefictions marks:",prediction)
linear_model = model.named_steps["linearregression"]
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)



