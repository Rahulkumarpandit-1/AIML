# Logistic Regression

## Concept

Logistic Regression is a **classification algorithm** used mainly for binary classification.

It predicts the **probability of a class**, then converts that probability into a class label.

Example:

`Hours studied → Pass / Fail`

- `0` → one class (e.g. Fail)
- `1` → another class (e.g. Pass)

The `0` and `1` are **class labels chosen in the dataset**. They do not inherently mean Fail or Pass.

## Important Terms

- **Class** → category the model predicts.
- **Binary classification** → two classes.
- **Probability** → model's confidence for each class.
- **Threshold** → boundary used to convert probability into a class.
- **Sigmoid** → converts the model's raw output into a value between 0 and 1.
- **Decision boundary** → separates different predicted classes.

Typical binary threshold:

`Probability ≥ 0.5 → Class 1`

`Probability < 0.5 → Class 0`

## Required Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
```

## Essential Syntax

```python
X = data[["feature"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

`predict()` → returns predicted class.

`predict_proba()` → returns probability for each class.

For binary classification:

```text
[class 0 probability, class 1 probability]
```

## Evaluation Metrics

### Accuracy
Percentage of all predictions that are correct.

`Accuracy = Correct predictions / Total predictions`

### Confusion Matrix

```text
              Predicted
              0      1
Actual  0     TN     FP
        1     FN     TP
```

- **TP** → actual 1, predicted 1
- **TN** → actual 0, predicted 0
- **FP** → actual 0, predicted 1
- **FN** → actual 1, predicted 0

### Precision

Of all predicted positives, how many were actually positive?

`Precision = TP / (TP + FP)`

### Recall

Of all actual positives, how many did the model correctly find?

`Recall = TP / (TP + FN)`

### F1-score

Balances Precision and Recall.

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

## Metric Syntax

```python
accuracy_score(y_test, predictions)

confusion_matrix(y_test, predictions)

precision_score(y_test, predictions)

recall_score(y_test, predictions)

f1_score(y_test, predictions)
```

## Core Flow

`Data → X/y → Train/Test Split → LogisticRegression → fit() → predict() → Evaluate`

For probabilities:

`predict_proba() → class probabilities → threshold/decision → class`

## Common Mistakes

- Thinking Logistic Regression is a regression algorithm for predicting continuous numbers.
- Confusing class labels with probabilities.
- Thinking `0` always means negative and `1` always means positive without checking the dataset.
- Passing `y_test` into `predict()`.
- Using MSE/RMSE as the main classification metrics.
- Assuming 100% accuracy on a tiny test set means the model is perfect.
- Confusing Precision with Recall.
- Forgetting that `predict_proba()` returns probabilities for **each class**.

## WORKFLOW
                 DATASET
                    ↓
              Separate X and y
                    ↓
             Train / Test Split
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
     X_train/y_train        X_test/y_test
          ↓                   ↓
       .fit()              keep hidden
          ↓
    MODEL LEARNS
          ↓
       X_test
          ↓
     .predict()
          ↓
     Predictions
          ↓
    Compare with y_test
          ↓
 ┌────────┼─────────┬──────────┐
 ↓        ↓         ↓          ↓
Accuracy  Confusion Precision  Recall
 Matrix             ↓          ↓
                     └────┬─────┘
                          ↓
                       F1-score