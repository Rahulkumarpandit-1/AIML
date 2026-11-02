# Reproducibility & Experiment Tracking

## 1. What Is Reproducibility?

Reproducibility means recording the data, code, settings, and environment needed to repeat an ML experiment and obtain consistent results.

It matters because ML results can change when data splits, random seeds, model settings, or software environments change.

Reproducibility improves:
- Trust in results.
- Debugging and collaboration.
- Model comparisons.
- Experiment auditing.
- Future maintenance.

## 2. Random Seeds

Many ML workflows use pseudorandom operations, including data splitting, initialization, and some training algorithms.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`random_state=42` makes the split repeatable under the same relevant conditions.

The number 42 is arbitrary. Any fixed seed can be used.

Important:
- A seed does not guarantee better performance.
- A seed controls only the operations that use it.
- Record seeds for randomized steps where supported.
- Some hardware and software operations can remain nondeterministic.

## 3. What Is an ML Experiment?

An experiment is a particular attempt to train and evaluate a model under defined conditions.

For example:

- Algorithm: Decision Tree
- `max_depth`: 5
- `min_samples_leaf`: 2
- Cross-validation: 5-fold stratified
- Metric: Accuracy
- Random seed: 42

Changing the hyperparameters or data split creates a different experiment that should be recorded separately.

## 4. What Should Be Recorded?

A useful experiment record includes:

| Field | Purpose |
|---|---|
| Experiment ID | Identifies the run |
| Timestamp | Shows when it was run |
| Dataset/version | Identifies the data used |
| Model | Identifies the algorithm |
| Hyperparameters | Records model settings |
| Random seed | Helps reproduce randomized operations |
| Validation method | Explains how the model was compared |
| Scoring metric | Defines what performance means |
| CV scores | Records validation performance |
| Test score | Records final held-out performance |
| Code version | Identifies the implementation |
| Environment | Records relevant package and software versions |

For more complex projects, also record preprocessing, feature selection, data-cleaning decisions, class weights, thresholds, and relevant artifacts.

## 5. Cross-Validation Scores

Suppose five validation folds produce:

`[0.91, 0.94, 0.92, 0.93, 0.90]`

Mean CV score:

Mean = sum of fold scores / number of folds

Mean = (0.91 + 0.94 + 0.92 + 0.93 + 0.90) / 5 = 0.92

The mean summarizes average performance across the folds.

The standard deviation summarizes how much the scores vary across folds. Higher variation may indicate that performance depends on the particular fold, although it is not by itself proof of instability.

## 6. Simple Experiment Tracking with CSV

A CSV log allows experiments to be compared without a specialized tracking tool.

```python
import csv

experiment = {
    "model": "DecisionTreeClassifier",
    "max_depth": 5,
    "random_state": 42,
    "cv_mean_accuracy": 0.92,
    "test_accuracy": 0.90
}

filename = "experiment_log.csv"

with open(filename, "a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=experiment.keys()
    )

    if file.tell() == 0:
        writer.writeheader()

    writer.writerow(experiment)
```

The numbers are illustrative. In a real workflow, populate the record with measured results.

This code appends a row to the CSV. The file is saved in the current working directory unless an explicit path is provided.

For long-running projects, use a stable schema for the log. If fields change between runs, manage the schema deliberately rather than assuming every row has identical columns.

## 7. Comparing Experiments Correctly

When comparing models:
1. Use the same dataset split where appropriate.
2. Use a consistent validation method.
3. Compare the same metric or explain why metrics differ.
4. Record hyperparameters and preprocessing.
5. Examine variability across CV folds.
6. Select using training/validation information.
7. Evaluate the final choice on the held-out test set.

A higher CV score does not guarantee a higher test score.

Avoid repeatedly selecting models based on test results. Doing so indirectly tunes your decisions to the test set.

## 8. Reproducibility and Data Leakage

Reproducibility does not automatically prevent leakage.

For example, fitting a scaler on the entire dataset before cross-validation leaks information from validation folds into the training process.

Use a pipeline when preprocessing learns from data so each fold fits its transformations only on the fold's training portion.

For final evaluation:
- Keep the test set separate during tuning.
- Fit the selected workflow using training data.
- Evaluate it on the held-out test set after selection.

## 9. Tools for Experiment Tracking

### CSV or spreadsheet

Good for small learning projects. Easy to inspect and compare.

### MLflow

Can track parameters, metrics, artifacts, and model versions. Useful as projects become more complex.

### Version control

Git records code changes. A commit hash can link an experiment to the exact code version that produced it.

Git alone does not automatically record every dataset, metric, or model artifact. Keep experiment records alongside version control.

## 10. Common Mistakes

1. Saving metrics without the corresponding hyperparameters.
2. Forgetting which dataset or split produced a result.
3. Changing the evaluation metric between experiments without noting it.
4. Treating a fixed seed as a guarantee of identical results in every environment.
5. Using test scores repeatedly to choose models.
6. Recording only the mean CV score while ignoring variation.
7. Failing to record preprocessing and package versions.
8. Assuming that recording an experiment makes the experiment methodologically sound.

## 11. Interview Questions

**Q1. Why use a random seed?**

To make supported randomized operations repeatable under the same conditions.

**Q2. What is experiment tracking?**

Recording configurations, metrics, data references, and other details so experiments can be compared and reproduced.

**Q3. Why record hyperparameters alongside metrics?**

Without the settings, it can be difficult to identify or reproduce the model that produced a score.

**Q4. Does a fixed random seed guarantee identical results everywhere?**

No. Software versions, hardware, and nondeterministic operations can still affect results.

**Q5. Why should we record CV scores and test scores separately?**

CV scores support model selection; the held-out test score provides a final evaluation of the selected workflow.

## Key Takeaways

- Use fixed seeds where appropriate to make experiments repeatable.
- Record data, preprocessing, hyperparameters, metrics, and relevant software versions.
- Use CSV logs for small projects and consider dedicated tracking tools as complexity grows.
- Compare experiments using consistent evaluation procedures.
- Keep test data out of model selection.
- Reproducibility helps you trust, debug, explain, and maintain ML work.