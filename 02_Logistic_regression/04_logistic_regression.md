# Logistic Regression

## 1. What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm primarily used for classification. It predicts the probability that an input belongs to a class.

Example: Predict whether a student passes (`1`) or fails (`0`).

## 2. Linear Score

The model first calculates a score:

\[
z=w_1x_1+w_2x_2+b
\]

Plain text: `z = w1*x1 + w2*x2 + b`

- `x`: input feature
- `w`: learned weight
- `b`: learned bias
- `z`: raw score

## 3. Sigmoid Function

The sigmoid converts the score into a probability between 0 and 1.

\[
p=\frac{1}{1+e^{-z}}
\]

Plain text: `p = 1 / (1 + e^(-z))`

- If `z = 0`, then `p = 0.5`.
- Positive scores produce probabilities above 0.5.
- Negative scores produce probabilities below 0.5.

## 4. Classification Threshold

By default, the threshold is 0.5:

- If `p >= 0.5`, predict class `1`.
- If `p < 0.5`, predict class `0`.

Changing the threshold changes class predictions, not the probability estimated by the already-trained model.

A lower threshold usually increases recall; a higher threshold often increases precision. These are tendencies, not guarantees.

## 5. Binary Cross-Entropy (Log Loss)

The loss measures how well the predicted probability matches the actual binary label.

\[
L=-[y\ln(p)+(1-y)\ln(1-p)]
\]

Plain text: `L = -[y*ln(p) + (1-y)*ln(1-p)]`

- `y`: actual label (`0` or `1`)
- `p`: predicted probability of class `1`
- `ln`: natural logarithm

If `y = 1`, loss is `-ln(p)`. If `y = 0`, loss is `-ln(1-p)`.

Correct, confident predictions have low loss. Wrong, confident predictions have high loss.

Training adjusts the weights and bias to minimize the average loss.

## 6. Training and Prediction with Scikit-learn

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

- `fit()`: learns model parameters from training data.
- `predict()`: returns predicted class labels.
- `predict_proba()`: returns probabilities for each class, in the model's class order.
- `StandardScaler`: standardizes numerical features.
- `Pipeline`: applies preprocessing and prediction consistently.

## 7. Evaluation Metrics

### Accuracy

\[
\text{Accuracy}=\frac{\text{Correct predictions}}{\text{Total predictions}}
\]

### Confusion Matrix

For binary classification, with class labels ordered as `0, 1`:

\[
\begin{bmatrix}
TN & FP\\
FN & TP
\end{bmatrix}
\]

- `TN`: true negative
- `FP`: false positive
- `FN`: false negative
- `TP`: true positive

### Precision

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

Of all predicted positives, how many were actually positive?

### Recall

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

Of all actual positives, how many did the model identify?

### F1-score

\[
F1=2\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
\]

F1 balances precision and recall.

- Macro average: treats every class equally.
- Weighted average: weights each class by its number of actual samples.
- Support: number of actual examples in each class.

## 8. Important Practical Points

- Use a validation set to choose the classification threshold.
- Evaluate the final model on an untouched test set.
- Small test sets produce unreliable performance estimates.
- Accuracy alone can be misleading when classes are imbalanced.
- Logistic Regression is a classification algorithm despite its name.
- Scaling is particularly useful when features have very different scales.

## 9. Quick Summary

Logistic Regression calculates a linear score, converts it into a probability using sigmoid, and applies a threshold to predict a class. Binary cross-entropy is commonly used as its training loss. Accuracy, precision, recall, F1-score, and the confusion matrix help evaluate its predictions.