# Machine Learning Fundamentals

## 1. What Is Machine Learning?

Machine Learning (ML) is a field of AI in which systems learn patterns from data to make predictions or decisions without being explicitly programmed for every possible case.

**Example:** A model learns from house data to predict the price of a new house.

### Main Types of Machine Learning

- **Supervised learning:** Learns from labeled data.
  - Regression: predicts numerical values.
  - Classification: predicts categories.
- **Unsupervised learning:** Finds patterns in unlabeled data.
  - Examples: clustering and dimensionality reduction.
- **Reinforcement learning:** An agent learns through actions, rewards, and penalties.

## 2. Samples, Features, Labels, and Targets

- **Sample:** One individual data record.
- **Feature:** An input variable used for prediction.
- **Label/Target:** The value the model is expected to predict.

Example: House-price prediction.

| Size (sq. ft.) | Bedrooms | Price |
|---:|---:|---:|
| 1,000 | 2 | ₹30 lakh |
| 1,500 | 3 | ₹45 lakh |
| 2,000 | 4 | ₹60 lakh |

Here, each row is a sample, size and bedrooms are features, and price is the target.

In scikit-learn, `X` usually represents features and `y` represents the target.

## 3. Training, Validation, and Test Sets

Data is commonly divided into three sets:

- **Training set:** Used to fit the model.
- **Validation set:** Used to compare models and tune hyperparameters.
- **Test set:** Used for final evaluation on unseen data.

The test set should not guide repeated model or hyperparameter choices.

For a simple split, 70% training, 15% validation, and 15% test may be reasonable, but the appropriate proportions depend on the dataset and task.

## 4. Parameters vs. Hyperparameters

### Parameters

Values learned by the model during training.

Example: The weights and intercept in linear regression.

### Hyperparameters

Settings chosen before or around training that control how the model learns.

Examples:
- Learning rate
- Number of epochs
- Tree depth
- Regularization strength

**Difference:** Parameters are learned from data; hyperparameters control the learning process or model configuration.

## 5. Predictions, Errors, and Residuals

A **prediction** is the output produced by a model.

For regression:

\[
e_i = y_i-\hat{y}_i
\]

Where:
- \(y_i\): Actual value
- \(\hat{y}_i\): Predicted value
- \(e_i\): Residual

Example: Actual price = ₹60 lakh; predicted price = ₹68 lakh.

\[
e=60-68=-8\text{ lakh}
\]

The negative residual means the model overpredicted the price.

A residual is signed. Error metrics such as MAE and MSE summarize errors across multiple predictions.

## 6. Loss Functions and Objective Functions

A **loss function** measures how wrong a model's prediction is, usually for one sample or a batch.

A common regression loss is squared error:

\[
L_i=(y_i-\hat{y}_i)^2
\]

An **objective function** is the quantity being optimized. It may combine average prediction loss with regularization penalties.

For example:

\[
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

This is the mean squared error objective for a regression model.

The goal of training is generally to minimize the chosen objective.

## 7. Empirical Risk Minimization (ERM)

Empirical Risk Minimization means choosing model parameters to minimize the average loss measured on the training dataset.

\[
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}L_i(\theta)
\]

Where:
- \(n\): Number of training samples
- \(L_i\): Loss for sample \(i\)
- \(\theta\): Model parameters

ERM is a core principle behind many supervised learning algorithms.

However, minimizing training loss alone does not guarantee good performance on unseen data.

## 8. Gradient Descent

Gradient descent is an optimization algorithm that updates model parameters in the direction that reduces the objective function.

\[
\theta_{\text{new}}=\theta_{\text{old}}-\eta\nabla J(\theta)
\]

Where:
- \(\theta\): Model parameters
- \(\eta\): Learning rate
- \(\nabla J(\theta)\): Gradient of the objective

The gradient indicates the direction of steepest increase locally. Subtracting it moves the parameters in the opposite direction.

### Simple Example

Suppose:

\[
J(\theta)=(\theta-3)^2
\]

Its gradient is:

\[
\frac{dJ}{d\theta}=2(\theta-3)
\]

If \(\theta=0\) and \(\eta=0.1\):

\[
\theta_{\text{new}}=0-0.1[-6]=0.6
\]

The parameter moves toward the minimum at \(\theta=3\).

## 9. Learning Rate

The learning rate controls the size of each parameter update.

- **Too small:** Learning can be slow.
- **Appropriate:** Updates can make steady progress.
- **Too large:** Updates may overshoot the minimum or fail to converge.

The best learning rate depends on the objective, model, and optimizer.

## 10. Epochs, Batches, and Iterations

- **Epoch:** One complete pass through the training dataset.
- **Batch:** A subset of training samples processed together.
- **Iteration:** One optimizer update.

If a dataset has 1,000 samples and the batch size is 100, one epoch contains 10 batches and typically 10 optimizer updates.

### Gradient Descent Variants

- **Batch Gradient Descent:** Uses the full training dataset for each update.
- **Stochastic Gradient Descent (SGD):** Uses one sample per update.
- **Mini-batch Gradient Descent:** Uses a small batch per update; widely used in practice.

## 11. Optimization Algorithms

### 11.1 Momentum

Momentum uses a moving average of gradients to help updates maintain useful movement.

\[
v_t=\beta v_{t-1}+g_t
\]

\[
\theta_{t+1}=\theta_t-\eta v_t
\]

Here, \(g_t\) is the current gradient and \(\beta\) controls how much previous movement is retained.

### 11.2 RMSProp

RMSProp tracks a moving average of squared gradients and adjusts the effective step size for each parameter.

\[
s_t=\rho s_{t-1}+(1-\rho)g_t^2
\]

\[
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t}+\epsilon}g_t
\]

- \(s_t\): Moving average of squared gradients
- \(\rho\): Decay rate
- \(\epsilon\): Small constant for numerical stability

Large recent squared gradients increase the denominator, generally reducing the effective update size.

### 11.3 Adam

Adam combines momentum-like gradient averaging with RMSProp-like squared-gradient averaging.

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

Because the moving averages start at zero, Adam applies bias correction:

\[
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
\]

\[
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
\]

The parameter update is:

\[
\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

Typical starting values are \(\beta_1=0.9\) and \(\beta_2=0.999\), although settings may need adjustment.

### Optimizer Summary

| Optimizer | Main idea |
|---|---|
| Gradient Descent | Uses the gradient to update parameters |
| Momentum | Incorporates previous gradients |
| RMSProp | Adapts updates using squared-gradient averages |
| Adam | Combines gradient averaging and adaptive scaling |

No optimizer is guaranteed to be best for every problem.

## 12. Underfitting and Overfitting

### Underfitting

The model fails to learn important patterns.

Typical signs:
- High training error
- High validation/test error

Possible remedies include using a more suitable model, adding useful features, or reducing excessive regularization.

### Overfitting

The model learns training-specific details, including noise, and performs poorly on unseen data.

Typical signs:
- Very low training error
- Significantly higher validation/test error

Possible remedies include regularization, simpler models, more representative data, and appropriate model selection.

## 13. Generalization

Generalization is a model's ability to perform well on unseen data.

For example, similar training and test performance may indicate good generalization, provided the evaluation is appropriate and the data is representative.

A small performance gap alone does not guarantee a good model: both training and test performance could be poor.

## 14. Bias–Variance Trade-off

### Bias

Bias is error associated with overly restrictive assumptions in a model.

High bias is commonly associated with underfitting.

### Variance

Variance describes how sensitive a model's learned behavior is to changes in the training dataset.

High variance is commonly associated with overfitting.

| | High bias | High variance |
|---|---|---|
| Typical behavior | Too simple | Too sensitive to training data |
| Training error | Often high | Often low |
| Generalization | Often poor | Often poor |
| Common issue | Underfitting | Overfitting |

The goal is to choose a model with suitable complexity that captures useful patterns and generalizes well.

## 15. Regularization and Model Complexity

Model complexity describes how flexible a model is in representing patterns.

Regularization adds a penalty to the training objective to discourage certain forms of model complexity.

\[
J_{\text{regularized}}(\theta)
=J_{\text{original}}(\theta)+\lambda\Omega(\theta)
\]

Where:
- \(J_{\text{original}}\): Original objective
- \(\Omega(\theta)\): Regularization penalty
- \(\lambda\): Regularization strength

### L1 Regularization (Lasso)

\[
J=\text{MSE}+\lambda\sum_j|w_j|
\]

L1 can drive some weights exactly to zero, making it useful for feature selection in some settings.

### L2 Regularization (Ridge)

\[
J=\text{MSE}+\lambda\sum_jw_j^2
\]

L2 penalizes large weights and generally shrinks them toward zero.

As regularization strength increases:
- Too little regularization may allow overfitting.
- An appropriate amount may improve generalization.
- Too much may cause underfitting.

Regularization strength is commonly selected using validation data or cross-validation.

## 16. Key Takeaways

1. ML models learn patterns from data to make predictions or decisions.
2. Features are inputs; targets are what the model predicts.
3. Training data fits the model, validation data guides choices, and test data supports final evaluation.
4. Parameters are learned; hyperparameters control model configuration or training.
5. Loss measures prediction error; an objective is what training optimizes.
6. Gradient descent updates parameters using gradients.
7. Momentum, RMSProp, and Adam modify how parameter updates are calculated.
8. Underfitting means failing to learn important patterns; overfitting means learning training-specific details.
9. Generalization measures performance on unseen data.
10. Bias, variance, model complexity, and regularization help explain and control model behavior.

**Roadmap position:** Part 03 — Machine Learning Fundamentals. The next major section is Part 05 — Supervised Machine Learning, since Data Preprocessing & Feature Engineering has already been studied.
