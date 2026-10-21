# Class Imbalance & Threshold Selection

## 1. What Is Class Imbalance?

Class imbalance occurs when one class has many more observations than another.

Examples:

- Fraud detection: 95% legitimate transactions, 5% fraudulent transactions.
- Disease screening: 98% negative cases, 2% positive cases.
- Spam detection: Most emails are legitimate, while fewer are spam.

The majority class has many observations, while the minority class has relatively few.

This can make learning and evaluation difficult because a model may perform well on the majority class but fail to identify the minority class.

## 2. Why Accuracy Can Be Misleading

Suppose a fraud dataset contains 1,000 transactions:

- 950 legitimate transactions.
- 50 fraudulent transactions.

A model predicts every transaction as legitimate.

Its accuracy is:

Accuracy = Correct predictions / Total predictions

Accuracy = 950 / 1,000 = **95%**

That looks impressive, but the model catches zero fraudulent transactions.

Its fraud recall is 0%.

This is why accuracy should not be used alone for imbalanced classification.

## 3. Metrics to Monitor

For a binary classifier, the confusion matrix contains:

- TP: Actual positive, predicted positive.
- TN: Actual negative, predicted negative.
- FP: Actual negative, predicted positive.
- FN: Actual positive, predicted negative.

### Precision

Precision = TP / (TP + FP)

Plain text: Of the cases predicted positive, what fraction were actually positive?

High precision means fewer false alarms among positive predictions.

### Recall

Recall = TP / (TP + FN)

Plain text: Of all actual positive cases, what fraction did the model identify?

High recall means fewer missed positive cases.

### F1-score

F1 = 2 × Precision × Recall / (Precision + Recall)

Plain text: The harmonic mean of precision and recall.

F1 is useful when both precision and recall matter. It does not directly account for true negatives.

### Which metric should we prioritize?

- Fraud detection where missed fraud is costly: often prioritize recall, while monitoring false alarms.
- Spam filtering where incorrectly blocking legitimate email is costly: precision may be especially important.
- Medical screening: recall may be important for the initial screening stage, with follow-up tests controlling false positives.
- Balanced concern for precision and recall: F1 can be useful.

The correct choice depends on the costs of different errors and the operational requirements.

## 4. What Is a Classification Threshold?

Many classifiers produce a probability for the positive class.

Example:

```text
Transaction A: 0.92
Transaction B: 0.61
Transaction C: 0.38
Transaction D: 0.08
```

A threshold converts these probabilities into class labels.

With threshold 0.50:

```text
Probability >= 0.50  -> class 1
Probability <  0.50  -> class 0
```

The predictions become:

```text
A -> 1
B -> 1
C -> 0
D -> 0
```

The threshold is a decision rule. It does not retrain the model or change the underlying probabilities.

## 5. How Changing the Threshold Affects Predictions

Consider two thresholds:

- Threshold 0.50: Predict positive when probability is at least 0.50.
- Threshold 0.20: Predict positive when probability is at least 0.20.

The lower threshold labels more observations as positive.

Generally, lowering the threshold:

- Increases or maintains the number of positive predictions.
- Increases or maintains recall.
- Increases or maintains false positives.
- Can reduce precision.

Raising the threshold generally has the opposite effects.

These are monotonic tendencies for a fixed set of scores; precision itself does not have to change in one direction at every threshold.

**Important:** A lower threshold does not make the model inherently smarter. It changes the balance between missed positives and false alarms.

## 6. Choosing a Threshold

The default threshold of 0.50 is not always appropriate, particularly for imbalanced datasets.

A practical procedure is:

1. Train the model on the training set.
2. Obtain predicted probabilities for the validation set.
3. Evaluate precision, recall, F1, or another metric at several candidate thresholds.
4. Choose a threshold that fits the task's objective and constraints.
5. Freeze the model and threshold.
6. Evaluate once on the test set to estimate final performance.

Do not repeatedly choose the threshold based on test results. That leaks information from the test set into model selection.

### Example threshold comparison

Suppose validation results are:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.38 | 0.38 | 0.38 |
| 0.30 | 0.36 | 0.19 | 0.25 |
| 0.40 | 0.50 | 0.095 | 0.16 |
| 0.50 | 1.00 | 0.048 | 0.091 |

Among these tested thresholds, 0.20 has the highest F1 and recall. If the objective is to balance precision and recall, it is the strongest candidate from this table.

However, a threshold should not be selected by F1 automatically if the business objective is different. If missed fraud is extremely costly, a recall constraint may be more appropriate. If investigators can review only a limited number of alerts, precision or the number of positive predictions may be more important.

## 7. Class Weights

Class weights make mistakes on certain classes more costly during model training.

For example, a classifier can assign a higher weight to the minority class so that errors involving that class have more influence on the training objective.

In scikit-learn, many classifiers support:

```python
class_weight="balanced"
```

This commonly adjusts class weights inversely to class frequencies.

Example:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)
```

Class weights can improve minority-class recall, but they do not guarantee it. They may also increase false positives. Evaluate their effect using validation data.

## 8. Resampling Techniques

Resampling changes the class distribution of the training data.

### Oversampling

Increase the number of minority-class training examples. Random oversampling can duplicate existing observations; methods such as SMOTE generate synthetic examples.

Potential benefit: The model receives more minority-class examples during training.

Potential risk: Duplicating examples can encourage overfitting, and synthetic examples may not always be realistic.

### Undersampling

Reduce the number of majority-class training examples.

Potential benefit: Less training data may be needed, and the classes become more balanced.

Potential risk: Useful information from majority-class observations may be discarded.

### Avoid data leakage

Split the data first. Apply oversampling or undersampling only to the training data, ideally inside a pipeline during cross-validation.

Do not oversample the entire dataset before splitting. That can leak information across training and validation/test sets.

Also remember that resampling changes the training distribution and may affect how well predicted probabilities reflect real-world class prevalence. Evaluate calibration when probability accuracy matters.

## 9. Complete Python Example: Threshold Evaluation

This example assumes that a classifier has already been trained. It uses validation probabilities to compare thresholds and test probabilities only for final evaluation.

```python
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Model is already trained.
# Validation and test data were kept separate from training.

val_prob = model.predict_proba(X_val)[:, 1]

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

# Choose this threshold using validation results.
best_threshold = 0.20  # Example only; justify using your results.

# Evaluate the frozen choice on the test set.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nFinal test confusion matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nFinal test classification report:")
print(classification_report(y_test, test_pred, zero_division=0))
```

`X_val`, `y_val`, `X_test`, and `y_test` must come from your existing data split. The value `0.20` is only an example; use the threshold justified by your validation results.

## 10. Understanding Your Imbalanced Classification Experiment

Your experiment produced:

```text
Brier score: 0.0529

Confusion matrix:
[[376   2]
 [ 22   0]]

Accuracy: 0.94
```

Scikit-learn orders the binary confusion matrix as:

```text
[[TN, FP],
 [FN, TP]]
```

Therefore:

- TN = 376
- FP = 2
- FN = 22
- TP = 0

The model correctly identified 376 negatives and incorrectly flagged 2 negatives as positive. However, it missed all 22 actual positive cases.

This gives:

- Accuracy = 376 / 400 = 94%
- Recall = 0 / 22 = 0%
- Precision = 0 / (0 + 2) = 0%

The 94% accuracy is misleading because the model identifies no positive cases at all.

### What happened with the threshold?

Your final test predictions used a threshold of 0.50. Your validation experiment showed that a threshold of 0.20 achieved a better F1-score among the thresholds tested.

These are two separate steps:

1. **Threshold loop:** Compare candidate thresholds using validation probabilities.
2. **Final test evaluation:** Apply the chosen threshold to test probabilities.

If the code still uses 0.50 for the final test predictions, it has not applied the threshold selected from validation.

Do not assume that choosing 0.20 guarantees better test performance. Apply the frozen choice to the test set once and report the actual result.

### What does the Brier score mean?

The Brier score for binary probability predictions is:

Brier score = (1/n) × Σ(pᵢ − yᵢ)²

Plain text: Average the squared differences between each predicted probability and its actual binary outcome.

A lower score is generally better. It evaluates the quality of probabilistic predictions, not just thresholded labels. It is affected by calibration and discrimination, so it should not be interpreted as a pure calibration measure.

## 11. Common Mistakes

1. **Trusting accuracy alone:** A model can achieve high accuracy by predicting only the majority class.
2. **Assuming 0.50 is always optimal:** The best threshold depends on the task and error costs.
3. **Changing the threshold and retraining unnecessarily:** Threshold selection changes the decision rule; it does not itself retrain the model.
4. **Choosing a threshold on test data:** This creates optimistic evaluation and weakens the independence of the test set.
5. **Resampling before splitting:** This can cause leakage and unrealistic evaluation.
6. **Assuming class weights solve everything:** They can change the error trade-off but may not improve every metric.
7. **Assuming high recall is always best:** Very high recall can come with an unacceptable number of false positives.
8. **Ignoring probability calibration:** A threshold based on probabilities is easier to interpret when those probabilities are well calibrated.

## 12. Interview Revision

**Q1. Why can accuracy be misleading with imbalanced classes?**

Because the majority class can dominate the score even if the model fails to detect the minority class.

**Q2. What happens when the classification threshold decreases?**

The model generally predicts more positive cases, increasing or maintaining recall and potentially increasing false positives.

**Q3. What is the difference between class weights and threshold selection?**

Class weights influence the training objective. Threshold selection changes how the model's scores are converted into class labels.

**Q4. Why should the threshold be chosen using validation data?**

To select a decision rule without using the final test set to tune the model.

**Q5. What does zero recall mean?**

The model failed to identify any actual positive cases.

**Q6. Should we always use SMOTE or oversampling?**

No. Compare approaches using appropriate validation and check for leakage, overfitting, and changes in probability calibration.

## Key Takeaways

- Imbalanced data can make accuracy misleading.
- Precision measures the reliability of positive predictions; recall measures how many actual positives are found.
- F1 summarizes precision and recall through their harmonic mean.
- Threshold selection controls the trade-off between false positives and false negatives.
- Class weights and resampling affect training; thresholds affect decision-making.
- Choose the threshold on validation data and evaluate the frozen choice on test data.
- Consider the real-world cost of errors rather than blindly maximizing one metric.