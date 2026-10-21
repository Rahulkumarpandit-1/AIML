# ROC-AUC & Precision–Recall Curves

## 1. Why Do We Need These Curves?

In binary classification, models such as Logistic Regression often produce probabilities rather than final class labels.

Example:

```text
Actual class:       1
Predicted probability of class 1: 0.82
```

To convert probabilities into class labels, we choose a threshold. With a threshold of 0.5, a probability of 0.82 becomes class 1.

However, a different threshold can produce different predictions, changing precision, recall, false positives, and false negatives.

**ROC and Precision–Recall curves help us evaluate a classifier across multiple thresholds.**

They are especially useful when we need to compare models or understand the trade-off between different types of classification errors.

## 2. Understanding the Confusion Matrix

For binary classification:

- TP (True Positive): Actual positive, predicted positive.
- TN (True Negative): Actual negative, predicted negative.
- FP (False Positive): Actual negative, predicted positive.
- FN (False Negative): Actual positive, predicted negative.

Two important metrics are:

Recall = TP / (TP + FN)

Plain text: Of all actual positive cases, how many did the model identify?

False Positive Rate (FPR) = FP / (FP + TN)

Plain text: Of all actual negative cases, how many did the model incorrectly identify as positive?

ROC and PR curves use these metrics differently.

## 3. ROC Curve

ROC stands for Receiver Operating Characteristic.

A ROC curve plots:

- **X-axis:** False Positive Rate (FPR)
- **Y-axis:** True Positive Rate (TPR), which is the same as recall.

As the classification threshold changes, the FPR and TPR change. Each threshold produces a point on the curve.

### How to interpret a ROC curve

- A curve closer to the **top-left corner** generally indicates better discrimination.
- The diagonal line represents random-ranking performance.
- A curve closer to the diagonal indicates weaker discrimination.
- A point at (0, 1) represents zero false positives and 100% true positive rate.

### Threshold trade-off

Lowering the threshold generally classifies more observations as positive:

- Recall often increases.
- False positives often increase too.

Raising the threshold generally classifies fewer observations as positive:

- False positives often decrease.
- Recall may decrease.

These trends are common, but the exact changes depend on the model's predicted scores and the dataset.

## 4. ROC-AUC

AUC means Area Under the Curve. ROC-AUC is the area under the ROC curve.

It summarizes how well the model ranks positive examples above negative examples across possible thresholds.

### Interpretation

- AUC = 1.0: Perfect ranking of positives above negatives.
- AUC = 0.5: Random-ranking performance, on average.
- AUC below 0.5: Ranking is worse than random; reversing the score direction would improve it.

AUC is not the percentage of predictions that are correct.

### Ranking interpretation

ROC-AUC can be interpreted as the probability that a randomly selected positive observation receives a higher model score than a randomly selected negative observation, with ties conventionally receiving half credit.

For example, ROC-AUC = 0.90 indicates strong ranking ability under this interpretation. It does not guarantee 90% accuracy at a particular threshold.

### Important limitation

ROC-AUC can look strong when the positive class is rare, even when the number of false positives is operationally unacceptable. Always consider the class distribution and the cost of errors.

## 5. Precision–Recall Curve

A Precision–Recall (PR) curve plots:

- **X-axis:** Recall
- **Y-axis:** Precision

Precision = TP / (TP + FP)

Plain text: Of all observations predicted positive, how many were actually positive?

Recall = TP / (TP + FN)

Plain text: Of all actual positives, how many did the model find?

### How to interpret it

A model that maintains high precision at high recall is often desirable.

- High precision: Fewer false positives among predicted positives.
- High recall: Fewer actual positives are missed.
- Increasing recall can come at the cost of lower precision.

The best trade-off depends on the task. For example, fraud detection may tolerate more false alarms to catch more fraud, while an expensive manual investigation process may require higher precision.

## 6. Why PR Curves Are Important for Imbalanced Data

Suppose a dataset has 1,000 observations:

- 990 negatives
- 10 positives

A classifier that predicts every observation as negative achieves 99% accuracy but finds no positive cases.

This shows why accuracy alone can be misleading.

ROC analysis considers the false positive rate relative to all actual negatives. With many negatives, a small FPR may still mean many false-positive alerts.

Precision directly reflects the proportion of predicted positives that are correct. This makes the PR curve especially informative when the positive class is rare.

**Rule of thumb:** For rare-positive problems such as fraud detection, disease screening, or anomaly detection, examine the PR curve and precision/recall at realistic operating thresholds—not ROC-AUC alone.

The baseline precision of a no-skill classifier is related to the positive-class prevalence. In the example above, that prevalence is 10 / 1,000 = 1%.

## 7. ROC-AUC vs PR-AUC

| Property | ROC curve / ROC-AUC | Precision–Recall curve / PR-AUC |
|---|---|---|
| X-axis | False Positive Rate | Recall |
| Y-axis | True Positive Rate / Recall | Precision |
| Focus | Positive detection relative to negative detection | Quality of positive predictions and positive detection |
| Useful for | General ranking discrimination | Particularly informative with rare positive classes |
| Main caution | May hide an operationally large number of false positives | Depends on positive-class prevalence and the chosen PR-AUC convention |

PR-AUC is not uniquely defined by every implementation. Average Precision (AP), which scikit-learn provides, summarizes the precision–recall curve using a particular weighting of precision values. Do not assume AP and every numerical integration method for PR-AUC are identical.

## 8. Python Implementation

Assume the data has already been split into training and test sets and the classifier has been trained.

```python
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# Probability of the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# ROC curve and ROC-AUC
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision–Recall curve and Average Precision
precision, recall, pr_thresholds = precision_recall_curve(
    y_test, y_prob
)
average_precision = average_precision_score(y_test, y_prob)

plt.figure()
plt.plot(
    recall,
    precision,
    label=f"Average Precision = {average_precision:.3f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.show()
```

`X_train`, `X_test`, `y_train`, and `y_test` must come from your dataset. If your model is already trained, use it rather than fitting another model.

### Important details

- `predict_proba(X_test)[:, 1]` extracts the probability for class 1.
- ROC and PR curves need scores or probabilities, not only thresholded class labels.
- `roc_thresholds` and `pr_thresholds` contain candidate thresholds associated with the curve calculations.
- The PR curve arrays can have different lengths: the returned thresholds correspond to the curve's threshold intervals.
- For an imbalanced problem, inspect precision and recall at practical thresholds rather than relying solely on a single summary score.

## 9. Common Mistakes

1. **Confusing ROC-AUC with accuracy:** ROC-AUC measures ranking ability across thresholds, not the fraction of correct predictions at one threshold.
2. **Using predicted labels to plot curves:** Use probabilities or decision scores so the threshold can vary.
3. **Ignoring class imbalance:** ROC-AUC may not tell the whole story when positives are rare.
4. **Assuming AUC chooses the threshold:** AUC summarizes performance across thresholds; it does not automatically select the best operating threshold.
5. **Treating high AUC as proof of good calibration:** A model can rank observations well but produce poorly calibrated probabilities.
6. **Tuning repeatedly on the test set:** Use validation data to select models and thresholds. Reserve the test set for final evaluation.

## 10. Interview Revision

**Q1. What does ROC-AUC measure?**

The model's ability to rank positive observations above negative observations across possible thresholds.

**Q2. What are the axes of the ROC curve?**

X-axis: False Positive Rate. Y-axis: True Positive Rate (recall).

**Q3. Why is the PR curve useful for fraud detection?**

It highlights the trade-off between catching fraud and ensuring that flagged transactions are actually fraudulent, which matters when fraud is rare.

**Q4. What does ROC-AUC = 0.5 mean?**

Random-ranking performance on average, assuming a meaningful binary ranking evaluation.

**Q5. Does ROC-AUC = 0.95 mean accuracy is 95%?**

No. Accuracy depends on the chosen threshold and the data distribution.

**Q6. Can a model have a high ROC-AUC and poor precision at the selected threshold?**

Yes. Ranking quality and precision at one operating threshold are different properties.

## Key Takeaways

- ROC plots TPR against FPR.
- ROC-AUC summarizes ranking discrimination across thresholds.
- PR plots precision against recall.
- PR analysis is particularly informative when the positive class is rare.
- Use probabilities or decision scores to construct these curves.
- Choose thresholds using validation data and report final performance on held-out test data.