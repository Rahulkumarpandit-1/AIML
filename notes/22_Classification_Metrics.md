# Classification Metrics — Complete Notes

## 1. Why Do We Need Classification Metrics?

Classification metrics measure how well a model predicts categorical labels.

Examples:
- Spam detection: spam or not spam.
- Disease detection: positive or negative.
- Fraud detection: fraudulent or legitimate.
- Image classification: cat, dog, or another category.

A model's accuracy alone may not reveal its weaknesses. We need multiple metrics to understand the kinds of errors it makes.

## 2. Confusion Matrix

A confusion matrix compares actual labels with predicted labels.

For binary classification, assume:
- Positive class = 1
- Negative class = 0

| Actual outcome | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | True Positive (TP) | False Negative (FN) |
| Actual negative | False Positive (FP) | True Negative (TN) |

### Definitions

- **True Positive (TP):** The actual class is positive, and the model predicts positive.
- **True Negative (TN):** The actual class is negative, and the model predicts negative.
- **False Positive (FP):** The actual class is negative, but the model predicts positive. This is a false alarm.
- **False Negative (FN):** The actual class is positive, but the model predicts negative. This is a missed positive.

In scikit-learn, for binary labels ordered as `[0, 1]`, the confusion matrix is arranged as:

`[[TN, FP], [FN, TP]]`

## 3. Worked Numerical Example

Suppose a disease detection model evaluates 100 people:

| Actual outcome | Prediction | Count |
|---|---|---:|
| Positive | Positive (TP) | 40 |
| Positive | Negative (FN) | 10 |
| Negative | Positive (FP) | 5 |
| Negative | Negative (TN) | 45 |
| **Total** | | **100** |

Therefore:

- TP = 40
- FN = 10
- FP = 5
- TN = 45

### 3.1 Accuracy

Accuracy measures the proportion of all predictions that are correct.

**Formula:**

Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Plain text:**

`Accuracy = (TP + TN) / Total`

Calculation:

`(40 + 45) / 100 = 0.85 = 85%`

Interpretation: 85% of the predictions were correct.

**Limitation:** Accuracy can be misleading when one class is much more common than the other.

### 3.2 Precision

Precision measures how many predicted positives are actually positive.

**Formula:**

Precision = TP / (TP + FP)

**Plain text:**

`Precision = TP / (TP + FP)`

Calculation:

`40 / (40 + 5) = 0.8889 = 88.89%`

Interpretation: Of all cases predicted positive, approximately 88.89% were actually positive.

Precision is important when false positives are costly, such as when legitimate transactions are incorrectly flagged as fraud.

### 3.3 Recall (Sensitivity or True Positive Rate)

Recall measures how many actual positive cases the model successfully identifies.

**Formula:**

Recall = TP / (TP + FN)

**Plain text:**

`Recall = TP / (TP + FN)`

Calculation:

`40 / (40 + 10) = 0.80 = 80%`

Interpretation: The model detected 80% of the actual positive cases.

Recall is important when missing a positive case is costly, such as in initial disease screening.

### 3.4 Specificity (True Negative Rate)

Specificity measures how many actual negative cases are correctly identified.

**Formula:**

Specificity = TN / (TN + FP)

**Plain text:**

`Specificity = TN / (TN + FP)`

Calculation:

`45 / (45 + 5) = 0.90 = 90%`

Interpretation: The model correctly identified 90% of actual negative cases.

### 3.5 F1-Score

F1-score is the harmonic mean of precision and recall.

**Formula:**

F1 = 2 × Precision × Recall / (Precision + Recall)

Calculation:

`2 × 0.8889 × 0.80 / (0.8889 + 0.80) ≈ 0.8421`

F1-score ≈ **84.21%**

F1 is useful when both precision and recall matter. It does not directly incorporate true negatives.

## 4. Which Metric Should We Choose?

| Situation | Metrics to consider | Reason |
|---|---|---|
| Balanced classification with similar error costs | Accuracy | Overall correctness is informative |
| Spam detection | Precision | Avoid incorrectly flagging legitimate emails |
| Initial disease screening | Recall | Reduce missed positive cases |
| Fraud detection | Precision, recall, F1 | Balance false alarms and missed fraud |
| Imbalanced classification | Precision, recall, PR-AUC | Accuracy may hide poor minority-class performance |
| Rejecting legitimate cases is costly | Specificity and precision | Measure negative-class detection and positive-prediction reliability |

Metric selection should reflect the real cost of each error.

## 5. Class Imbalance and the Accuracy Trap

Consider a dataset with 1,000 transactions:
- 990 legitimate transactions
- 10 fraudulent transactions

A model predicts every transaction as legitimate.

It achieves:

Accuracy = 990 / 1000 = 99%

But it detects zero fraudulent transactions.

Its fraud recall is 0%, making it useless for finding fraud despite its high accuracy.

This is why accuracy should not be used alone for highly imbalanced classification.

## 6. Macro, Micro, and Weighted Averages

In multiclass classification, precision, recall, and F1 can be calculated separately for each class and then combined.

### Macro average

Calculate the metric for each class, then take the unweighted mean.

Every class receives equal weight, regardless of its size.

Useful when performance on minority classes matters as much as performance on majority classes.

### Micro average

Aggregate the relevant counts across classes before calculating the metric.

This gives more influence to classes with more examples. In ordinary single-label multiclass classification, micro precision, micro recall, and micro F1 equal accuracy.

### Weighted average

Calculate the metric for each class, then average using each class's support (number of actual examples) as its weight.

This reflects class frequency, so majority classes contribute more.

### Support

Support is the number of actual examples belonging to each class.

A classification report typically includes precision, recall, F1-score, and support.

## 7. Scikit-Learn Implementation

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred))
```

Expected confusion matrix:

```text
[[4 1]
 [1 4]]
```

Therefore:
- TN = 4
- FP = 1
- FN = 1
- TP = 4

Accuracy = 8/10 = 0.80

Precision = 4/5 = 0.80

Recall = 4/5 = 0.80

F1-score = 0.80

### Important scikit-learn details

- Binary precision, recall, and F1 use class `1` as the positive class by default.
- `confusion_matrix()` uses the order of the supplied labels; by default, labels are sorted.
- `classification_report()` displays per-class results and macro and weighted averages.
- If there are no predicted positives, precision is undefined mathematically. Scikit-learn provides configurable handling through `zero_division`.

## 8. Common Mistakes

1. Using accuracy as the only metric for imbalanced data.
2. Confusing precision with recall.
3. Forgetting which class is positive.
4. Ignoring false positives and false negatives.
5. Comparing F1 scores without checking which averaging method was used.
6. Treating a high validation score as proof of strong real-world performance.
7. Choosing a threshold based on the test set and then reporting performance on that same test set as if it were untouched.

## 9. Interview Questions

**Q1. What is the difference between precision and recall?**

Precision measures the proportion of predicted positives that are correct. Recall measures the proportion of actual positives that are detected.

**Q2. Why can accuracy be misleading?**

A model can predict only the majority class and achieve high accuracy while failing to detect the minority class.

**Q3. What does F1-score measure?**

It combines precision and recall using their harmonic mean.

**Q4. What is a false negative?**

An actual positive example incorrectly predicted as negative.

**Q5. What is specificity?**

The proportion of actual negatives correctly identified: `TN / (TN + FP)`.

**Q6. What is the difference between macro and weighted averages?**

Macro averaging treats every class equally. Weighted averaging accounts for the number of actual examples in each class.

## Key Takeaway

A good evaluation should explain not only how often a model is correct, but also which errors it makes. Use accuracy for overall correctness when appropriate, precision for false-positive concerns, recall for false-negative concerns, and F1 when balancing precision and recall is useful.
