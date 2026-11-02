# Calibration and Error Analysis

## 1. What Is Probability Calibration?

Probability calibration measures whether a model's predicted probabilities agree with the actual frequency of outcomes.

Suppose a fraud detection model assigns 100 transactions probabilities close to 0.80.

If the model is well calibrated, approximately 80 of those transactions should be fraudulent over the long run.

| Predicted probability | Actual positive frequency | Interpretation |
|---|---|---|
| 0.80 | 80% | Well calibrated |
| 0.80 | 45% | Overconfident |
| 0.80 | 95% | Underconfident |

These are illustrative examples.

**Important distinction:**
- Accuracy measures the proportion of correct class predictions.
- ROC-AUC measures ranking or discrimination ability.
- Calibration measures the reliability of predicted probabilities.

A model can have good ROC-AUC but poor calibration.

## 2. Calibration Curve

A calibration curve compares:
- X-axis: average predicted probability in each bin.
- Y-axis: observed proportion of positive outcomes in that bin.

A perfectly calibrated model follows the diagonal line \(y=x\).

Interpretation:
- Curve near the diagonal: good calibration.
- Curve below the diagonal: observed positive frequency is lower than predicted; the model tends to overpredict.
- Curve above the diagonal: observed positive frequency is higher than predicted; the model tends to underpredict.

The curve is estimated from finite data, so small deviations may occur due to sampling noise.

## 3. Brier Score

The Brier score measures the mean squared difference between predicted probabilities and actual binary outcomes.

Formula:

\[
BS=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

Plain text:

`Brier score = average of (predicted probability - actual label)^2`

Where:
- \(N\) = number of observations.
- \(p_i\) = predicted probability of class 1.
- \(y_i\) = actual label, either 0 or 1.

### Numerical example

Suppose the actual labels are:

`y = [1, 0, 1]`

Predicted probabilities:

`p = [0.8, 0.3, 0.4]`

Squared errors:

- First: \((0.8-1)^2=0.04\)
- Second: \((0.3-0)^2=0.09\)
- Third: \((0.4-1)^2=0.36\)

Therefore:

\[
BS=\frac{0.04+0.09+0.36}{3}
=\frac{0.49}{3}\approx0.1633
\]

A lower Brier score is generally better for probabilistic predictions. However, it reflects both calibration and discrimination/resolution, so it is not a pure calibration measure.

## 4. What Is Error Analysis?

Error analysis is the process of examining incorrect predictions to understand why a model fails.

Instead of asking only, "What is my model's accuracy?", ask:
- Which examples are misclassified?
- Are false negatives more common than false positives?
- Are mistakes concentrated in certain classes or subgroups?
- Are input features missing, noisy, or misleading?
- Is the model overfitting?
- Does the evaluation data differ from the training data?

### Example: Fraud detection

Suppose a model misses many fraudulent transactions.

Possible causes:
1. Fraud examples are rare.
2. Fraudulent transactions resemble legitimate transactions.
3. Relevant features are missing.
4. The classification threshold is too high.
5. The training data does not represent newer fraud patterns.

Each cause suggests a different solution. Lowering the threshold may increase recall, but it may also increase false positives.

## 5. Confusion Matrix for Error Analysis

For binary classification, scikit-learn uses:

`[[TN, FP], [FN, TP]]`

- TN: legitimate transactions correctly identified.
- FP: legitimate transactions incorrectly flagged as fraud.
- FN: fraudulent transactions missed.
- TP: fraudulent transactions correctly detected.

Recall focuses on missed positive cases:

\[
Recall=\frac{TP}{TP+FN}
\]

Precision focuses on the reliability of positive predictions:

\[
Precision=\frac{TP}{TP+FP}
\]

These metrics help describe errors, but investigating individual examples is often necessary to understand their causes.

## 6. Python Implementation

File: `machine_learning/07_model_evaluation/05_calibration_error_analysis.py`

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("Brier score:", brier_score_loss(y_test, prob))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification report:")
print(classification_report(
    y_test, pred, zero_division=0
))

CalibrationDisplay.from_predictions(
    y_test, prob, n_bins=10, strategy="uniform"
)
plt.title("Calibration Curve")
plt.show()

# Inspect the first ten test examples
for i in range(min(10, len(y_test))):
    if pred[i] != y_test[i]:
        print(
            f"Index: {i}, Actual: {y_test[i]}, "
            f"Predicted: {pred[i]}, "
            f"Probability: {prob[i]:.3f}"
        )
```

### Important evaluation rule

The example uses the test set to demonstrate calibration plots and error inspection. In a rigorous model-selection workflow:
- Fit the model on training data.
- Fit calibration methods and make model-selection decisions using training/validation data or cross-validation.
- Reserve the test set for final evaluation.

Do not repeatedly modify your model based on test-set errors.

## 7. Common Mistakes

1. Assuming high accuracy means good calibration.
2. Assuming a high ROC-AUC guarantees reliable probabilities.
3. Treating the Brier score as a pure calibration measure.
4. Changing the model repeatedly based on test-set results.
5. Lowering the classification threshold without considering false-positive costs.
6. Investigating only aggregate scores and never examining individual errors.

## 8. Interview Questions

**Q1. What is probability calibration?**

Whether predicted probabilities correspond to observed outcome frequencies.

**Q2. Can a model have good ROC-AUC but poor calibration?**

Yes. It may rank positive examples above negative examples while predicting probabilities that are systematically too high or too low.

**Q3. What does the Brier score measure?**

The mean squared difference between predicted probabilities and actual binary labels. Lower is generally better.

**Q4. What is error analysis?**

Investigating model mistakes to identify patterns, causes, and opportunities for improvement.

**Q5. Why should we avoid repeatedly tuning against the test set?**

Because decisions become influenced by test-set information, making the final evaluation less trustworthy.

## 9. Completion Checklist

- [ ] Understand calibration versus discrimination.
- [ ] Understand calibration curves and the diagonal reference.
- [ ] Calculate a Brier score manually.
- [ ] Run the Python implementation.
- [ ] Inspect the calibration curve and misclassified examples.
- [ ] Complete the checkpoint questions.

**Status:** Notes prepared. Practical execution and experiments remain unconfirmed.
