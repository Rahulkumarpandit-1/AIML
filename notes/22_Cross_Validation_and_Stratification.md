# Cross-Validation and Stratification

## 1. Why Cross-Validation?

A single train-validation split can give an unreliable estimate because performance depends on which samples happen to be selected.

Cross-validation evaluates a model across multiple splits and provides a more stable estimate of its performance.

**Example:** Five validation accuracies are 90%, 92%, 88%, 91%, and 89%.

Mean score = (90 + 92 + 88 + 91 + 89) / 5 = 90%.

The mean summarizes performance across folds. The standard deviation measures how much scores vary across folds.

## 2. Training, Validation, and Test Sets

- **Training set:** used to fit model parameters.
- **Validation set:** used for model selection, hyperparameter tuning, and other development decisions.
- **Test set:** used for final evaluation after development decisions are complete.

With cross-validation, the training portion is divided into folds. Each fold takes a turn serving as validation data.

**Important:** Do not repeatedly tune a model against the test set. This can cause information from the test set to influence model selection, making the final score overly optimistic.

## 3. K-Fold Cross-Validation

In K-Fold cross-validation:

1. Divide the available training data into K approximately equal folds.
2. Use K − 1 folds for training.
3. Use the remaining fold for validation.
4. Repeat until every fold has been used for validation.
5. Aggregate the K validation scores.

For 5-fold cross-validation, the model is trained and evaluated five times.

A common choice is K = 5 or K = 10, although the appropriate value depends on dataset size, computational cost, and the problem.

## 4. Stratified K-Fold

Stratified K-Fold tries to preserve class proportions across folds.

Example dataset:

- 900 negative samples
- 100 positive samples

Positive-class proportion = 100 / 1000 = 10%.

Stratification attempts to maintain this proportion in each fold.

It is especially useful for imbalanced classification. It does not guarantee identical proportions in every fold, especially when classes are extremely small.

For time-series data, ordinary shuffled K-Fold may be inappropriate because it can allow future observations to influence validation of earlier observations.

## 5. Common Cross-Validation Methods

### KFold
Divides samples into K folds. It is commonly used for regression and general-purpose cross-validation when its assumptions fit the data.

### StratifiedKFold
Preserves class proportions as closely as possible. Commonly used for classification.

### GroupKFold
Keeps samples from the same group in the same fold. Useful when multiple rows belong to the same person, patient, customer, or device.

### TimeSeriesSplit
Creates sequential training and validation splits that respect temporal order. Useful for time-series forecasting and other time-dependent tasks.

Choose the split method based on how the data was collected and how the model will be used.

## 6. Implementing Cross-Validation

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

# Keep a separate test set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("Fold scores:", scores)
print("Mean CV accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Fit on the complete training set after model selection
pipeline.fit(X_train, y_train)

print("Test accuracy:", pipeline.score(X_test, y_test))
```

### Understanding the output

- `n_splits=5`: creates five folds.
- `shuffle=True`: shuffles samples before dividing them into folds.
- `random_state=42`: makes the shuffled splits reproducible.
- `cross_val_score()`: returns one score per fold.
- `scores.mean()`: calculates the average fold score.
- `scores.std()`: measures score variability.
- `pipeline.fit()`: fits the final pipeline on the complete training set.
- `pipeline.score()`: evaluates that fitted pipeline on the held-out test set.

Exact scores may vary with the dataset, library versions, and configuration.

## 7. Why Use a Pipeline?

Preprocessing steps such as scaling, imputation, feature selection, and dimensionality reduction may learn information from the data.

Incorrect approach:

1. Scale the entire dataset.
2. Perform cross-validation.

The scaler has already used information from samples that will later serve as validation data. This is a form of data leakage.

Correct approach:

1. Put preprocessing and the estimator inside a pipeline.
2. Pass the pipeline to `cross_val_score()`.
3. Let each fold fit preprocessing using only its training portion.

This preserves the separation between training and validation data.

## 8. `cross_val_score()` vs. `cross_validate()`

- `cross_val_score()` returns one evaluation score per fold.
- `cross_validate()` can return multiple metrics and additional timing information. It can also return fitted estimators or train scores when configured.

Example:

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print(results["test_accuracy"])
print(results["test_f1"])
```

For binary classification, the default positive class used by common binary precision, recall, and F1 scorers is generally class 1. Confirm the positive class and scoring requirements for your task.

## 9. Advantages

- Uses multiple splits instead of relying on one validation split.
- Helps compare candidate models more reliably.
- Reveals variation in performance across folds.
- Supports model selection and hyperparameter tuning.
- Can make efficient use of limited training data.

## 10. Limitations

- Requires multiple training runs and can be computationally expensive.
- Does not eliminate all uncertainty or guarantee good performance on future data.
- Poorly chosen splits can still produce misleading results.
- Related samples, temporal order, or duplicated records may require specialized splitting.
- Repeatedly selecting models using the same cross-validation results can overfit the validation process.

## 11. Common Mistakes

1. Scaling or selecting features using the entire dataset before cross-validation.
2. Using the test set repeatedly to select models.
3. Using ordinary shuffled folds for time-dependent data.
4. Allowing related samples to appear in both training and validation folds.
5. Interpreting the mean score without checking variability.
6. Assuming stratification guarantees equal class counts in every fold.
7. Treating a high cross-validation score as proof that the model will generalize well.

## 12. Interview Questions

**Q1. Why is cross-validation useful?**

It estimates model performance across multiple data splits and reduces dependence on one particular validation split.

**Q2. Why use StratifiedKFold?**

It attempts to preserve class proportions across folds, which is particularly useful for imbalanced classification.

**Q3. Why keep a separate test set?**

To evaluate the final selected model on data that was not used for fitting or model-selection decisions.

**Q4. How does a pipeline help prevent leakage?**

It ensures learned preprocessing steps are fitted separately within each training fold.

**Q5. When would GroupKFold be appropriate?**

When samples belong to groups that must not be split between training and validation, such as multiple records from the same patient.

**Q6. When would TimeSeriesSplit be appropriate?**

When chronological order matters and validation must simulate predicting later observations from earlier data.

## Key Takeaway

Cross-validation supports reliable model selection, stratification helps preserve class proportions, and pipelines keep learned preprocessing inside each fold. The final test set should remain untouched until the model-development decisions are complete.
