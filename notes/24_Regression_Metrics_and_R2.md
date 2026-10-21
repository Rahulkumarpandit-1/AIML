# Regression Metrics & R²

## 1. Why Do We Need Regression Metrics?

Regression models predict continuous numerical values, such as house prices, salaries, temperatures, or sales.

After training a model, we need to measure how far its predictions are from the actual values.

Let:

- `y` = actual value
- `ŷ` (y-hat) = predicted value
- `n` = number of observations
- `e = y - ŷ` = prediction error or residual

A residual's sign tells us whether the prediction was too low or too high:

- Positive residual: actual value is greater than prediction; the model underpredicted.
- Negative residual: actual value is less than prediction; the model overpredicted.

## 2. Numerical Example

Suppose the actual values are `[3, -0.5, 2, 7]` and predictions are `[2.5, 0, 2, 8]`.

| Actual | Predicted | Error (`y - ŷ`) | Squared error |
|---:|---:|---:|---:|
| 3 | 2.5 | 0.5 | 0.25 |
| -0.5 | 0 | -0.5 | 0.25 |
| 2 | 2 | 0 | 0 |
| 7 | 8 | -1 | 1 |

The absolute errors are `[0.5, 0.5, 0, 1]`.

Different regression metrics summarize these errors in different ways.

## 3. Mean Absolute Error (MAE)

MAE calculates the average absolute prediction error.

**Formula:**

MAE = (1/n) × Σ |yᵢ − ŷᵢ|

Plain text: Add the absolute errors and divide by the number of observations.

For our example:

MAE = (0.5 + 0.5 + 0 + 1) / 4 = **0.5**

### Interpretation

The predictions are off by an average absolute amount of 0.5 units.

- MAE is easy to interpret.
- It uses the same units as the target.
- It is less sensitive to large errors than MSE.
- It does not indicate whether errors are consistently too high or too low.

Use MAE when you want an understandable measure of typical prediction error and prefer not to penalize large errors disproportionately.

## 4. Mean Squared Error (MSE)

MSE calculates the average of the squared prediction errors.

**Formula:**

MSE = (1/n) × Σ (yᵢ − ŷᵢ)²

Plain text: Square every error, add the squared errors, and divide by the number of observations.

For our example:

MSE = (0.25 + 0.25 + 0 + 1) / 4 = **0.375**

### Why square the errors?

Squaring prevents positive and negative errors from cancelling each other out. It also penalizes large errors more heavily.

For example:

- Error of 2 → squared error of 4
- Error of 5 → squared error of 25

An error that is 2.5 times larger contributes 6.25 times as much squared error.

### Advantages and limitations

- Useful when large mistakes should receive strong penalties.
- Commonly used as a model training loss.
- Sensitive to outliers because large errors are squared.
- Expressed in squared target units, which can be harder to interpret.

## 5. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

**Formula:**

RMSE = √[(1/n) × Σ (yᵢ − ŷᵢ)²]

Plain text: Calculate MSE, then take its square root.

For our example:

RMSE = √0.375 ≈ **0.6124**

### Why use RMSE?

RMSE has the same units as the target, making it easier to interpret than MSE. It still penalizes large errors more strongly than MAE.

For the same observations, RMSE is at least as large as MAE.

Use RMSE when large prediction errors are particularly undesirable and you want the result expressed in the original target units.

## 6. MAE vs MSE vs RMSE

| Metric | What it measures | Outlier sensitivity | Units |
|---|---|---|---|
| MAE | Average absolute error | Lower than MSE/RMSE in its penalty on large errors | Target units |
| MSE | Average squared error | High | Squared target units |
| RMSE | Square root of average squared error | High | Target units |

For all three metrics, **lower is generally better**, when evaluating models on the same target and dataset.

Do not compare raw MSE or RMSE values across targets measured on different scales without considering those scales.

## 7. R² — Coefficient of Determination

R² compares the model's squared prediction errors with the errors made by a simple baseline that always predicts the mean of the actual target values.

**Formula:**

R² = 1 − (SS_res / SS_tot)

Where:

- `SS_res = Σ(yᵢ − ŷᵢ)²`: residual sum of squares, measuring model errors.
- `SS_tot = Σ(yᵢ − ȳ)²`: total sum of squares, measuring variation around the actual target mean.
- `ȳ` = mean of the actual target values.

Plain text:

R² = 1 − (model's squared errors / squared errors from predicting the target mean)

### Interpreting R²

- **R² = 1:** Perfect predictions; residual sum of squares is zero.
- **R² = 0:** The model has the same squared-error performance as the mean-prediction baseline on this evaluated dataset.
- **R² < 0:** The model performs worse than that baseline on this evaluated dataset.

R² does not directly represent prediction accuracy or the percentage of predictions that are correct.

An R² of 0.80 does not mean 80% of individual predictions are correct.

On an ordinary test dataset, R² can be negative. It is not generally restricted to the range 0 to 1.

## 8. Why R² Can Be Misleading

R² measures performance relative to a baseline; it does not tell you the typical error in the target's original units.

Consider two models:

- Model A: R² = 0.85, MAE = ₹50,000
- Model B: R² = 0.80, MAE = ₹20,000

Model B has lower MAE, while Model A has higher R². Which is preferable depends on the evaluation objective and the full error distribution.

Use R² alongside MAE or RMSE instead of relying on R² alone.

## 9. Python Implementation with Scikit-learn

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

Expected results, approximately:

```text
MAE: 0.5
MSE: 0.375
RMSE: 0.6124
R²: 0.9486
```

`root_mean_squared_error` is available in recent scikit-learn versions. If your installed version does not support it, use:

```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

## 10. Common Mistakes

1. **Choosing only R²:** A good R² does not guarantee small errors in real-world units.
2. **Confusing MSE with RMSE:** MSE uses squared units; RMSE returns to the target's units.
3. **Thinking R² cannot be negative:** A model can perform worse than the mean baseline on held-out data.
4. **Comparing metrics from different datasets:** Differences in target ranges and difficulty can make comparisons misleading.
5. **Evaluating on training data alone:** Training performance can hide overfitting. Use held-out data or appropriate cross-validation.
6. **Assuming MAE ignores outliers:** MAE still counts outliers; it simply does not square their errors.

## 11. Interview Revision

**Q1. Which metric penalizes large errors more strongly?**

MSE and RMSE, because they are based on squared errors.

**Q2. Which metric is easiest to interpret in the original target units?**

MAE and RMSE.

**Q3. What does R² = 0 mean?**

The model matches the mean-prediction baseline in squared-error performance on the evaluated dataset.

**Q4. Can R² be negative?**

Yes. This can happen when the model performs worse than the mean-prediction baseline.

**Q5. Should MAE or RMSE always be preferred?**

No. MAE is less sensitive to large errors; RMSE places more emphasis on them. Choose based on the consequences of different types of prediction errors.

## Key Takeaways

- MAE measures average absolute error.
- MSE measures average squared error.
- RMSE is the square root of MSE and uses the target's units.
- R² compares model performance with a mean-prediction baseline.
- Lower error metrics are generally better; R² closer to 1 is generally better.
- Evaluate regression models on data not used to fit them.