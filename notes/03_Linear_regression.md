# Linear Regression — Notes

## 1. What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical target, such as marks, salary, or house prices.

**Simple Linear Regression:**

\[
\hat{y}=wx+b
\]

- \(x\): input feature
- \(\hat{y}\): predicted value
- \(w\): learned weight or coefficient
- \(b\): learned intercept

**Multiple Linear Regression:**

\[
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

Each coefficient represents the expected change in the prediction for a one-unit change in its feature, holding other features fixed.

## 2. How does Linear Regression learn?

The model selects parameters that minimize prediction errors on the training data.

A common objective is Mean Squared Error (MSE):

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Scikit-learn's `LinearRegression` generally solves the least-squares problem directly rather than using the gradient-descent loop studied earlier.

## 3. Regression evaluation metrics

- **MAE:** Mean absolute prediction error. Lower is better.
- **MSE:** Mean squared error; penalizes large errors more strongly. Lower is better.
- **RMSE:** Square root of MSE, expressed in the target's units. Lower is better.
- **R²:** Measures performance relative to predicting the mean of the target. Higher is generally better; it can be negative.

R² is not the same as prediction accuracy percentage.

## 4. Train/test split

- Training data fits the model.
- Test data evaluates performance on held-out observations.
- A validation set or cross-validation can help select models and hyperparameters.

Evaluating only on training data does not establish how well a model generalizes to unseen data.

## 5. Polynomial Regression

Polynomial Regression adds powers of the original features, such as:

\[
\hat{y}=b+w_1x+w_2x^2
\]

It can model curved relationships. Increasing polynomial degree also increases model complexity and can cause overfitting.

## 6. Regularization

Regularization adds a penalty to the training objective to discourage overly large coefficients.

**Ridge — L2 penalty**

\[
J=MSE+\alpha\sum_j w_j^2
\]

Shrinks coefficients toward zero; generally does not set them exactly to zero.

**Lasso — L1 penalty**

\[
J=MSE+\alpha\sum_j|w_j|
\]

Can set coefficients exactly to zero, providing a form of feature selection.

**Elastic Net**

Combines L1 and L2 penalties. In scikit-learn, `alpha` controls overall regularization strength and `l1_ratio` controls the balance between L1 and L2.

A larger `alpha` means stronger regularization. Too much regularization can cause underfitting.

## 7. Feature scaling and pipelines

Regularized models are commonly used with standardized features because feature scales affect the penalty.

`StandardScaler` transforms features using training-set means and standard deviations. Fit it only on training data, then apply the same transformation to validation/test data.

A scikit-learn `Pipeline` keeps preprocessing and model fitting together and helps prevent data leakage.

## 8. Model vs optimization algorithm

A **model/objective** defines the relationship being learned and what is minimized.

An **optimization algorithm or solver** finds parameters that minimize that objective.

Examples of optimization methods include Gradient Descent, SGD, Momentum, RMSProp, and Adam. Ridge, Lasso, and Elastic Net are regularized regression models/objectives, not optimization algorithms.

## 9. Assumptions and limitations

Classical Linear Regression inference commonly relies on assumptions including linearity in parameters, independent errors, and constant error variance. Normally distributed errors are mainly relevant to certain statistical inference procedures.

Perfect multicollinearity prevents unique identification of individual coefficients. Strongly correlated features can also make coefficient interpretation unstable.

Always assess generalization with suitable held-out data or cross-validation.

## 10. Practical learning summary

1. Train the model with `fit()`.
2. Generate predictions with `predict()`.
3. Evaluate predictions using MAE, MSE, RMSE, and R².
4. Split data into training and test sets.
5. Compare model configurations using the same split and metrics.
6. Scale features appropriately for regularized models.
7. Select hyperparameters using validation data or cross-validation.
8. Never assume regularization improves every dataset.

**Current progress:** Completed core Linear Regression experiments covering simple/multiple regression, polynomial regression, Ridge, Lasso, Elastic Net, and model evaluation. Next topic: Logistic Regression.