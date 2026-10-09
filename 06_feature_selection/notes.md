# Feature Selection — Machine Learning Notes

## 1. What Is Feature Selection?

**Feature selection** is the process of choosing useful input features and removing features that are irrelevant, redundant, or less useful for a particular ML task.

- **Feature:** An input variable used by a model.
- **Target:** The value the model predicts.
- **Feature selection:** Chooses which existing features to retain.
- **Feature engineering:** Creates or transforms features.

Example:
- Features: `hours`, `attendance`, `age`
- Target: `passed`

### Why use feature selection?
- Remove irrelevant or redundant features.
- Reduce model complexity and training cost.
- Improve interpretability.
- Potentially reduce overfitting and improve generalization.

**Important:** Feature selection does not guarantee better accuracy. Removing useful information can reduce performance.

---

## 2. Correlation and Feature Relationships

**Correlation** measures the strength and direction of a linear relationship between two numerical variables.

The Pearson correlation coefficient is:

\[
r=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
\]

Where:
- \(r\) = correlation coefficient
- \(\operatorname{Cov}(X,Y)\) = covariance between variables
- \(\sigma_X,\sigma_Y\) = their standard deviations

Range: \(-1\) to \(+1\).

- \(+1\): Perfect positive linear relationship.
- \(0\): No linear correlation.
- \(-1\): Perfect negative linear relationship.

```python
df.corr(numeric_only=True)
```

### Two different questions

**Feature vs target:** Is the feature associated with the target?

**Feature vs feature:** Do two input features contain overlapping information?

### Multicollinearity

Multicollinearity occurs when predictor features are strongly linearly related to one another.

It can make coefficients unstable or difficult to interpret in models such as Linear Regression and Logistic Regression.

High correlation alone is not sufficient reason to remove a feature.

**Limitations:**
- Correlation does not prove causation.
- Zero correlation does not rule out a nonlinear relationship.
- Correlation with the target is a clue, not proof of predictive usefulness.

---

## 3. Variance Threshold

**Variance** measures how much a feature's values vary around its mean.

Population variance:

\[
\operatorname{Var}(X)=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

Where:
- \(n\) = number of observations
- \(x_i\) = an observed value
- \(\bar{x}\) = mean of the feature

Example:

- `[1, 1, 1, 1]` → variance is \(0\).
- `[1, 1, 1, 2]` → variance is greater than \(0\).

### Syntax

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0)
X_selected = selector.fit_transform(X)
```

- `fit()` learns which features meet the threshold.
- `transform()` removes features below the threshold.
- `fit_transform()` performs both operations.

With `threshold=0`, constant features are removed.

**Limitations:**
- It does not use the target `y`.
- Low variance does not automatically mean a feature is useless.
- Variance depends on the feature's scale.

---

## 4. Univariate Feature Selection — SelectKBest

**Univariate** means evaluating one feature at a time against the target.

`SelectKBest` scores each feature independently, ranks the scores, and retains the top \(k\) features.

### Syntax

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
```

- `score_func`: Statistical scoring function.
- `k`: Number of features to retain.
- `scores_`: Score for each input feature.
- `pvalues_`: Associated p-values when supported by the scoring function.
- `get_support()`: Boolean mask identifying retained features.

```python
print(selector.scores_)
print(selector.pvalues_)
print(selector.get_support())
print(X.columns[selector.get_support()])
```

### Common scoring functions

| Scoring function | Appropriate use |
|---|---|
| `f_classif` | Numerical features and a categorical classification target |
| `f_regression` | Numerical features and a numerical regression target |
| `chi2` | Non-negative features with a classification target |

### A. `f_classif`

`f_classif` performs an ANOVA F-test for each numerical feature against the classes of a classification target.

The F-statistic compares variation between class means with variation within classes.

Conceptually:

\[
F=\frac{\text{variation between classes}}{\text{variation within classes}}
\]

A larger F-statistic indicates stronger separation between class means relative to within-class variation.

**Statistical assumptions:** Observations are independent; the feature values within each class are approximately normally distributed for the classical test's interpretation; and class variances are reasonably similar.

Violations can affect the reliability of p-values. Feature selection should also be validated against actual predictive performance.

### B. `f_regression`

`f_regression` tests the linear relationship between each numerical feature and a numerical target.

For a simple linear relationship based on correlation \(r\), the F-statistic is:

\[
F=\frac{r^2}{1-r^2}(n-2)
\]

Where:
- \(r\) = feature-target Pearson correlation
- \(n\) = number of observations

The formula assumes the ordinary single-predictor test with \(n>2\) and \(|r|<1\). Perfect correlation is a limiting case.

A higher score indicates stronger evidence of a linear association, not necessarily better performance in every model.

### C. `chi2`

The chi-square test compares observed frequencies with expected frequencies under a null hypothesis of independence.

For frequency counts:

\[
\chi^2=\sum\frac{(O-E)^2}{E}
\]

Where:
- \(O\) = observed frequency
- \(E\) = expected frequency under independence

A larger statistic indicates a greater discrepancy between observed and expected frequencies.

In scikit-learn, `chi2` is commonly used with non-negative features such as counts or one-hot encoded categorical variables.

**Important limitations:**
- Features must be non-negative.
- Do not blindly apply it to arbitrary continuous measurements.
- The test's statistical interpretation depends on the underlying assumptions, including adequate expected frequencies for the usual approximation.

### Understanding p-values

A p-value measures how compatible an observed statistic is with the null hypothesis, assuming the test's assumptions hold.

- Small p-value: Evidence against the null hypothesis.
- Large p-value: Insufficient evidence against the null hypothesis.

A p-value is **not** the probability that a feature is useless or that the null hypothesis is true.

When many features are tested, multiple-testing effects can produce false positives. Statistical significance also does not guarantee predictive value.

### Example

Suppose the scores are:

| Feature | F-score |
|---|---:|
| `attendance` | 54 |
| `hours` | 37.5 |
| `age` | 13.5 |

With `k=2`, `SelectKBest` retains `attendance` and `hours`.

**Remember:** `k=2` means two features (columns), not two observations (rows). A feature excluded by a fixed `k` is not necessarily useless.

---

## 5. Recursive Feature Elimination — RFE

**RFE** stands for Recursive Feature Elimination.

It repeatedly fits an estimator, ranks features using the estimator's importance information, removes less-important features, and repeats until the requested number remains.

### Syntax

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

selector = RFE(
    estimator=LogisticRegression(),
    n_features_to_select=2,
    step=1
)

X_selected = selector.fit_transform(X, y)
```

- `estimator`: Model used to rank features.
- `n_features_to_select`: Number of features to retain.
- `step=1`: Remove one feature per elimination iteration.
- `support_`: Boolean mask of retained features.
- `ranking_`: Feature rankings; rank `1` means retained.

```python
print(selector.support_)
print(selector.ranking_)
print(X.columns[selector.support_])
```

If `step` is larger, RFE can remove multiple features per iteration. The default step is one feature when `step` is an integer defaulting to `1`.

### Limitations
- Results depend on the estimator and training data.
- Repeated model fitting can be computationally expensive.
- Correlated features can substitute for one another.
- RFE does not guarantee better test performance.

---

## 6. Model-Based Selection — SelectFromModel

`SelectFromModel` uses an estimator's learned feature importances or coefficients to select features.

Examples of estimators that expose suitable importance information include Decision Trees and many linear models.

### Syntax

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

selector = SelectFromModel(
    estimator=DecisionTreeClassifier(random_state=42),
    threshold="mean"
)

X_selected = selector.fit_transform(X, y)
```

- `estimator`: Model used to estimate importance.
- `threshold="mean"`: Uses the mean importance as the selection cutoff.
- `threshold_`: Actual fitted threshold.
- `get_support()`: Boolean mask of retained features.
- `estimator_.feature_importances_`: Fitted Decision Tree's feature-importance values.

### Example

Suppose the importances are:

- `hours`: `0.0`
- `attendance`: `1.0`
- `age`: `0.0`

The mean importance is:

\[
\text{Mean importance}=\frac{0+1+0}{3}=0.3333
\]

`SelectFromModel` normally retains features whose importance is greater than or equal to the threshold. Thus, `attendance` is retained.

### Important limitations
- Importance values depend on the estimator.
- Decision Tree importance can favor features with many possible split points or categories in some settings.
- A zero importance means the fitted model assigned no importance under its measure; it does not prove the feature is universally useless.
- Model-based selection can still overfit if feature selection is evaluated on the same data used to choose features.

---

## 7. Comparing Feature-Selection Methods

| Method | How it selects |
|---|---|
| `VarianceThreshold` | Removes features with insufficient variance |
| `SelectKBest` | Scores features individually and keeps the top \(k\) |
| `RFE` | Repeatedly removes features using an estimator |
| `SelectFromModel` | Keeps features meeting a model-based importance threshold |

### Choosing a method

- Constant features → consider `VarianceThreshold`.
- Individual statistical scoring → consider `SelectKBest`.
- Fixed number of features with iterative estimator-based ranking → consider RFE.
- Estimator-based importance threshold → consider `SelectFromModel`.

The best method depends on the data, target, estimator, and evaluation results.

---

## 8. Feature Selection and Data Leakage

**Data leakage** occurs when information that would not legitimately be available during training or prediction influences model training or evaluation.

Incorrect workflow:

`Full dataset → Feature selection → Train/test split`

Correct workflow:

`Dataset → Split → Fit selection on training data → Transform training/test data → Train model → Evaluate on test data`

Any data-dependent feature selection must be fitted using training data only.

### Leakage-safe Pipeline

A scikit-learn `Pipeline` keeps feature selection and model fitting together.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("selection", SelectKBest(score_func=f_classif, k=2)),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

This ensures the selector is fitted on `X_train, y_train` during training. When predicting, the already-fitted selector transforms `X_test` before the model predicts.

### Cross-validation

**Cross-validation** evaluates a model across multiple train/validation splits.

When using cross-validation, put feature selection inside the Pipeline. The selector is then refitted within each training fold rather than using information from the validation fold.

This prevents feature-selection leakage across folds.

**Important:** The final test set should remain untouched until final evaluation.

---

## 9. Practical Rules and Common Mistakes

1. More features do not automatically mean a better model.
2. Statistical association does not prove causation.
3. Correlation and feature importance are not universal measures of usefulness.
4. Univariate selection can miss interactions between features.
5. A low score does not automatically mean a feature is useless.
6. Fit feature selection only on training data.
7. Use a Pipeline when combining selection with model training and cross-validation.
8. Choose the scoring function based on the target type and data requirements.
9. Evaluate the full modeling workflow with appropriate validation data or cross-validation.
10. Small datasets can produce unstable feature rankings.
11. Avoid target leakage: don't use information unavailable at prediction time.
12. Compare results with and without feature selection; keep it only when it benefits the objective.

---

## 10. Essential Imports

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    chi2,
    RFE,
    SelectFromModel
)
from sklearn.pipeline import Pipeline
```

## Mental Model

- **VarianceThreshold:** Does the feature vary?
- **SelectKBest:** Which features score highest individually?
- **RFE:** Which features remain after repeated estimator-based elimination?
- **SelectFromModel:** Which features meet the model-importance threshold?
- **Pipeline:** How do we fit selection and prediction steps consistently without leakage?

**Final principle:** Feature selection is a technique to test and potentially improve a modeling workflow—not a guarantee that fewer features produce better predictions.

## ColumnTransformer

### Definition
`ColumnTransformer` applies different preprocessing transformations to different columns of a dataset.

### Why it is needed
Real-world datasets often contain both numerical and categorical features. Different feature types require different preprocessing methods.

### Common transformations
- Numerical features: `SimpleImputer` and `StandardScaler`
- Categorical features: `SimpleImputer` and `OneHotEncoder`

### OneHotEncoder
Converts categorical values into binary indicator columns. For example, Delhi becomes `[1, 0]` and Mumbai becomes `[0, 1]`.

This avoids introducing a false numerical ordering between categories.

### Pipeline vs ColumnTransformer
- `Pipeline`: chains sequential processing steps.
- `ColumnTransformer`: applies different transformations to specified columns.
- They are commonly combined: a `ColumnTransformer` handles column-specific preprocessing, while an outer `Pipeline` connects preprocessing to the model.

### Data leakage rule
Split the dataset first. Fit the complete pipeline on training data only. Use the fitted pipeline to transform and predict on test or production data.

### Unknown categories
`OneHotEncoder(handle_unknown="ignore")` prevents prediction from failing when a previously unseen category appears.

### Important distinction
`ColumnTransformer` does not automatically prevent every form of data leakage. Correct splitting and fitting practices are still required.