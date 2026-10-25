# Loss Functions

## 1. What is a loss function?

A loss function measures how different a model's prediction is from the correct answer.

During training, the model aims to reduce its loss by learning better parameter values.

Examples:
- Predicting house prices: regression
- Predicting whether a customer will churn: binary classification

## 2. Mean Squared Error (MSE)

MSE is commonly used for regression tasks where the model predicts numerical values.

Formula:

`MSE = (1/n) * sum((y - y_pred)^2)`

Where:
- `n` = number of examples
- `y` = actual value
- `y_pred` = predicted value

### Example

Actual house price = ₹10 lakh  
Predicted house price = ₹8 lakh

Error:

`10 - 8 = 2 lakh`

Squared error:

`(10 - 8)^2 = 4 lakh^2`

For one example, the MSE is 4 lakh-squared units. With multiple examples, MSE averages their squared errors.

### Key points

- Squaring prevents positive and negative errors from cancelling.
- Larger errors receive disproportionately more weight.
- MSE is sensitive to outliers because errors are squared.

## 3. Binary Cross-Entropy (BCE)

Binary cross-entropy is commonly used for binary classification when the model outputs a probability-like score for class 1.

Formula:

`L = -[y*ln(p) + (1-y)*ln(1-p)]`

Where:
- `y` = actual label, either 0 or 1
- `p` = predicted probability of class 1
- `ln` = natural logarithm

### When the actual label is 1

The formula simplifies to:

`L = -ln(p)`

Examples:

- `p = 0.90` → loss ≈ 0.105
- `p = 0.20` → loss ≈ 1.609
- `p = 0.01` → loss ≈ 4.605

The closer the predicted probability is to 1, the lower the loss when the actual label is 1.

A confident wrong prediction receives a large loss.

### When the actual label is 0

The formula simplifies to:

`L = -ln(1-p)`

The model receives lower loss when it assigns a low probability to class 1.

## 4. MSE versus Binary Cross-Entropy

| Feature | MSE | Binary Cross-Entropy |
|---|---|---|
| Common task | Regression | Binary classification |
| Input comparison | Numerical prediction vs actual value | Predicted probability vs actual label |
| Main behavior | Squares prediction errors | Penalizes incorrect probabilities |
| Common example | House price prediction | Customer churn prediction |

These are common choices, not universal rules. The appropriate loss depends on the task, model output, and training setup.

## 5. How loss connects to training

A typical neural-network training cycle is:

1. **Forward pass:** The model produces predictions.
2. **Loss calculation:** Compare predictions with the correct labels.
3. **Backpropagation:** Calculate gradients showing how parameters affect the loss.
4. **Optimizer step:** Update the model's parameters using those gradients.
5. Repeat across training batches and epochs.

### Important distinction

- Loss function: measures prediction error.
- Backpropagation: calculates gradients.
- Optimizer: uses gradients to update parameters.

Reducing training loss does not automatically guarantee good performance on unseen data. Validation and test evaluation are also necessary.

## 6. Key takeaways

1. Loss measures how wrong a model's predictions are.
2. MSE is a common choice for regression.
3. Binary cross-entropy is a common choice for binary classification with probability outputs.
4. Confident incorrect predictions can receive large cross-entropy losses.
5. The optimizer updates weights; the loss function itself does not.
6. Training aims to reduce loss while maintaining good generalization to unseen data.
