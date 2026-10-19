# Anomaly Detection

## 1. Definition

Anomaly detection identifies observations that differ significantly from the expected pattern of a dataset.

Examples include fraudulent transactions, faulty sensors, unusual network traffic, and abnormal measurements.

An anomaly is not automatically an error or a harmful event. It is an observation that deserves further investigation.

## 2. Classification vs. Anomaly Detection

- **Classification:** predicts known classes using labeled examples.
- **Anomaly detection:** identifies unusual observations, often without labeled anomaly examples.

Anomaly detection can be unsupervised, semi-supervised, or supervised, depending on the method and available labels.

## 3. Isolation Forest

Isolation Forest isolates observations through random feature selections and split points.

Unusual observations are often isolated in fewer splits than ordinary observations.

Important parameters:

- `n_estimators`: number of trees.
- `contamination`: assumed outlier proportion used to set the threshold.
- `random_state`: supports reproducibility.

Output labels:

- `1`: inlier.
- `-1`: predicted outlier.

`contamination` is a thresholding assumption, not proof of the true anomaly rate.

## 4. Local Outlier Factor (LOF)

LOF compares the local density of an observation with the density of its neighbors.

A point in a substantially lower-density region may receive an outlier score.

LOF is useful when anomalies are unusual relative to their local neighborhood, even if they are not globally far from all other points.

## 5. One-Class SVM

One-Class SVM learns a boundary around normal observations and flags points that fall outside the learned region.

Its performance depends on parameter choices, feature scaling, and the data's structure.

## 6. Evaluation

When ground-truth labels are available, use:

- **Precision:** proportion of flagged observations that are truly anomalous.
- **Recall:** proportion of actual anomalies that are detected.
- **F1-score:** balance between precision and recall.

Accuracy can be misleading when anomalies are rare.

Without labels, inspect flagged observations, use domain knowledge, and check whether results remain stable under reasonable parameter changes.

## 7. Practical Workflow

1. Inspect the data and understand the features.
2. Handle missing values and consider feature scaling where appropriate.
3. Fit an appropriate anomaly detection method.
4. Inspect flagged observations and anomaly scores.
5. Tune the threshold and parameters.
6. Evaluate with labeled examples if available.
7. Investigate anomalies before taking action.

## 8. Limitations

- Normal but rare observations may be flagged.
- Real anomalies may be missed.
- Results depend on parameters and feature representation.
- Some methods are sensitive to feature scales.
- Without labels, it is difficult to measure true detection quality.

## Key Takeaway

Anomaly detection flags unusual observations; it does not automatically determine whether they are fraudulent, faulty, or harmful. Isolation Forest is a useful starting point, while LOF and One-Class SVM provide alternative approaches.
