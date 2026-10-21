# Hyperparameter Tuning

## 1. What Is Hyperparameter Tuning?

Hyperparameter tuning is the process of testing different model settings to find a configuration that performs well on unseen data.

A model can use the correct algorithm and still perform poorly because its settings are unsuitable.

The goal is not to maximize training performance. The goal is to select settings that generalize well.

## 2. Parameters vs Hyperparameters

### Model parameters

Values learned from training data.

Examples:
- Weights and bias in Linear or Logistic Regression.
- Split features and thresholds learned by a Decision Tree.

### Hyperparameters

Settings chosen by the practitioner before or around model training.

Examples:
- KNN: `n_neighbors`
- Decision Tree: `max_depth`, `min_samples_leaf`
- Random Forest: `n_estimators`, `max_features`
- Logistic Regression: `C`
- SVM: `C`, `kernel`, `gamma`

**Remember:** Model training learns parameters; hyperparameter tuning searches for suitable settings.

## 3. Why Tuning Is Necessary

Consider illustrative Decision Tree results:

| Maximum depth | Training accuracy | Validation accuracy |
|---:|---:|---:|
| 2 | 90% | 88% |
| 5 | 99% | 95% |
| 15 | 100% | 91% |

- A shallow tree may underfit.
- A moderately sized tree may generalize better.
- A very deep tree may overfit the training data.

These results are examples, not guaranteed outcomes.

A good configuration is selected using validation or cross-validation performance, not training accuracy alone.

## 4. Grid Search

Grid Search tests every combination of the hyperparameter values specified in a grid.

Example:

```python
param_grid = {
    "max_depth": [2, 5, 10],
    "min_samples_split": [2, 10]
}
```

Number of combinations:

3 × 2 = 6

Each possible combination is evaluated. If 5-fold cross-validation is used, this normally requires:

6 combinations × 5 folds = 30 model fits.

The selected estimator is then typically refitted on all data supplied to the search when `refit=True`.

### Advantages

- Systematic and easy to understand.
- Evaluates every combination in the specified grid.
- Useful when the search space is relatively small.

### Limitations

- Can become computationally expensive.
- Results depend on the values included in the grid.
- A good value outside the grid will not be found.

## 5. Randomized Search

Randomized Search evaluates a specified number of sampled hyperparameter combinations.

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=15,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
```

`n_iter=15` requests 15 sampled combinations. When all parameters are supplied as finite lists, scikit-learn samples without replacement if fewer than all possible combinations are requested.

### Advantages

- Often faster for large search spaces.
- Can explore more varied ranges when distributions are supplied.
- Lets you control the number of combinations tested.

### Limitations

- May miss the best combination.
- Results depend on the sampled candidates.
- More iterations require more computation.

## 6. Grid Search vs Randomized Search

| Feature | Grid Search | Randomized Search |
|---|---|---|
| Search method | Every specified combination | A specified number of sampled combinations |
| Computational cost | Can be high | Can be controlled using `n_iter` |
| Best use case | Small, manageable search spaces | Large search spaces |
| Guarantee | Finds the best result among tested grid combinations, according to the chosen scoring and CV setup | Finds the best result among sampled combinations |
| Common API | `GridSearchCV` | `RandomizedSearchCV` |

Neither method guarantees the globally best possible model.

## 7. Cross-Validation During Tuning

Cross-validation provides a more stable estimate than relying on a single validation split.

For 5-fold cross-validation:

1. Divide the training data into five folds.
2. Train on four folds and validate on the fifth.
3. Repeat five times, using each fold for validation once.
4. Average the validation scores.
5. Compare candidate configurations using those averages.

For example, if a candidate receives validation accuracies of:

0.91, 0.94, 0.92, 0.93, 0.90

Mean CV accuracy = (0.91 + 0.94 + 0.92 + 0.93 + 0.90) / 5 = 0.92

This is an illustrative calculation.

For classification, `StratifiedKFold` attempts to preserve class proportions across folds. It is often a suitable choice for binary or multiclass classification.

## 8. Important GridSearchCV Attributes

After calling `search.fit(X_train, y_train)`:

- `search.best_params_`: best hyperparameter configuration found.
- `search.best_score_`: mean cross-validation score for that configuration.
- `search.best_estimator_`: selected estimator, refitted on the supplied training data by default when `refit=True`.
- `search.cv_results_`: detailed results for all evaluated candidates.

Example:

```python
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
print(search.best_estimator_)
```

The score's meaning depends on the `scoring` argument. If scoring is accuracy, `best_score_` is mean cross-validation accuracy. If scoring is F1, it is mean cross-validation F1.

## 9. Choosing the Scoring Metric

The search optimizes the metric specified in `scoring`.

Examples:

```python
scoring="accuracy"
scoring="f1"
scoring="precision"
scoring="recall"
scoring="roc_auc"
```

Choose a metric that matches the task.

- Accuracy: useful when overall correctness is appropriate and class imbalance is not hiding important failures.
- Precision: useful when false positive predictions are costly.
- Recall: useful when missing positive cases is costly.
- F1: balances precision and recall through their harmonic mean.
- ROC-AUC: evaluates ranking discrimination across thresholds.

A search optimizing accuracy will not necessarily select the configuration with the best recall or ROC-AUC.

## 10. A Safe Tuning Workflow

1. Split the data into training and test sets.
2. Keep the test set separate from model and hyperparameter selection.
3. Define candidate hyperparameters and an appropriate scoring metric.
4. Run Grid Search or Randomized Search with cross-validation on training data.
5. Select the configuration using cross-validation results.
6. Inspect the selected configuration and check for suspicious overfitting or unstable results.
7. Evaluate the selected estimator on the test set.
8. Report test performance honestly.

### Avoid data leakage

Do not use test-set performance to choose hyperparameters repeatedly.

If preprocessing learns from data—for example, imputation, scaling, or feature selection—place it inside a suitable scikit-learn pipeline so that each cross-validation fold learns those transformations only from its training portion.

If the data is time-dependent, ordinary random cross-validation may be inappropriate. Use a time-aware split that respects chronology.

## 11. Common Mistakes

1. Choosing hyperparameters using training accuracy alone.
2. Tuning repeatedly against the test set.
3. Searching a huge grid without considering computational cost.
4. Selecting the wrong scoring metric for the task.
5. Performing preprocessing before cross-validation when it learns statistics from all observations.
6. Assuming the best CV score must equal test performance.
7. Assuming a larger model or more complex configuration must be better.
8. Forgetting that random search can produce different sampled candidates if its random seed changes.

## 12. Interview Questions

**Q1. What is hyperparameter tuning?**

Searching for model settings that improve generalization performance.

**Q2. What is the difference between parameters and hyperparameters?**

Parameters are learned during training; hyperparameters are chosen or searched for by the practitioner.

**Q3. How does Grid Search differ from Randomized Search?**

Grid Search evaluates every specified combination. Randomized Search samples a limited number of combinations.

**Q4. Why use cross-validation during tuning?**

It compares candidate settings across multiple training/validation splits rather than relying on one split.

**Q5. What does `best_score_` mean?**

The mean cross-validation score for the selected candidate under the configured scoring metric.

**Q6. Should the test set be used for tuning?**

No. Keep it reserved for final evaluation after the configuration has been selected.

**Q7. Why can Randomized Search be more efficient?**

It can explore a large search space without evaluating every possible combination.

## Key Takeaways

- Hyperparameter tuning searches for settings that generalize well.
- Grid Search tests all specified combinations.
- Randomized Search samples a controlled number of combinations.
- Cross-validation helps compare configurations more reliably.
- Select a metric appropriate to the task.
- Keep preprocessing within the cross-validation pipeline when it learns from data.
- Never use the final test set repeatedly to tune a model.