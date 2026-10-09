# Linear Regression

## Concept
Linear Regression predicts a **continuous numerical value** by learning a linear relationship between input features and the target.

Example:
`Experience → Salary`

Basic equation:
`y = mx + b`

- `y` → predicted value
- `x` → feature
- `m` → slope/weight
- `b` → intercept/bias

## Important Terms

- **X (features)** → inputs used for prediction.
- **y (target)** → value the model predicts.
- **Training data** → used to learn model parameters.
- **Test data** → unseen data used to evaluate the trained model.
- **Generalization** → performing well on unseen data.

## Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

## Essential Syntax

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

## Model Parameters

```python
model.coef_
model.intercept_
```

- `coef_` → learned slope/weight.
- `intercept_` → predicted starting value when feature is 0.

## Evaluation

**MSE (Mean Squared Error)**
- Average of squared prediction errors.
- Larger errors are penalized more.
- Unit is squared.

**RMSE (Root Mean Squared Error)**
- `√MSE`
- Same unit as the target.
- Easier to interpret than MSE.

```python
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
```

## Prediction vs Evaluation

**Known actual target:**
`prediction + actual value → calculate error`

**New/unseen input:**
`input → prediction`

If the actual target is unknown, MSE/RMSE cannot be calculated yet.

## Visualization

- `scatter()` → shows actual data points.
- `plot()` → can show the regression line.

```python
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()
```

## Core Flow

`Data → X/y → Train/Test Split → Model → fit() → predict() → Evaluate → Visualize`

## Common Mistakes

- Training the model on test data.
- Passing `y_test` into `predict()`.
- Calculating MSE with mismatched numbers of actual and predicted values.
- Using evaluation metrics when the actual target is unknown.
- Confusing `coef_` with the prediction itself.