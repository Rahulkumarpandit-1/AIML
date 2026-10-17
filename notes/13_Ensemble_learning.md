# 13. Ensemble Learning

## 1. Introduction to Ensemble Learning

### 1.1 What is Ensemble Learning?

Ensemble Learning is a machine learning technique that combines predictions from multiple models to produce a final prediction.

Instead of relying on only one model, an ensemble uses several models and combines their outputs.

**Main objective:** Improve predictive performance, stability, or robustness.

For example, suppose three classifiers predict whether an email is spam:

- Model 1: Spam
- Model 2: Spam
- Model 3: Not Spam

Using majority voting, the ensemble predicts **Spam**.

### 1.2 Basic Diagram

```text
                 Input Data
                     |
          +----------+----------+
          |          |          |
       Model 1     Model 2    Model 3
          |          |          |
        Spam       Spam      Not Spam
          |          |          |
          +----------+----------+
                     |
              Voting / Averaging
                     |
               Final Prediction
                   Spam
```

### 1.3 Why use Ensemble Learning?

A single model may:
- Overfit the training data.
- Miss important patterns.
- Be sensitive to changes in the training dataset.
- Make incorrect predictions.

Combining models can reduce some of these problems.

However, an ensemble does not automatically perform better. If its models make similar errors, combining them may provide little improvement.

### 1.4 Important terminology

| Term | Meaning |
|---|---|
| Base model / base learner | An individual model inside an ensemble |
| Ensemble | The combined collection of models |
| Prediction aggregation | Combining model predictions |
| Diversity | Differences in the predictions or errors of models |
| Generalization | Performance on previously unseen data |
| Bias | Error caused by overly simplistic assumptions |
| Variance | Sensitivity of a model to changes in training data |
| Overfitting | Learning training-specific patterns that do not generalize |
| Hyperparameter | A setting chosen before or during model training, such as tree depth |

### 1.5 Three major ensemble approaches

```text
                ENSEMBLE LEARNING
                       |
        +--------------+--------------+
        |              |              |
      Bagging        Boosting       Stacking
        |              |              |
    Parallel        Sequential     Meta-model
    training        training      combines outputs
        |              |              |
    Random Forest   XGBoost      Model combination
```

- **Bagging:** Train models independently on different bootstrap samples.
- **Boosting:** Build an ensemble sequentially, with later stages improving the existing ensemble.
- **Stacking:** Train a separate model, called a meta-model, to combine predictions from other models.

---

## 2. Bagging — Bootstrap Aggregating

### 2.1 What is Bagging?

Bagging stands for **Bootstrap Aggregating**.

It creates multiple training datasets by sampling the original training data with replacement, trains separate models on these samples, and combines their predictions.

### 2.2 What is bootstrap sampling?

Bootstrap sampling means randomly selecting observations from a dataset **with replacement**.

With replacement means that after an observation is selected, it can be selected again.

Suppose the original dataset is:

```text
Original dataset:
[A, B, C, D, E]
```

Different bootstrap samples might be:

```text
Sample 1: [A, B, B, D, E]
Sample 2: [A, C, C, D, D]
Sample 3: [B, B, C, E, E]
```

Some observations appear more than once, while others may be absent from a particular sample.

### 2.3 Bagging workflow

```text
                  Training Dataset
                         |
              +----------+----------+
              |          |          |
          Bootstrap  Bootstrap  Bootstrap
           Sample 1   Sample 2   Sample 3
              |          |          |
           Model 1    Model 2    Model 3
              |          |          |
              +----------+----------+
                         |
                 Aggregate Results
                         |
                 Final Prediction
```

### 2.4 How are predictions combined?

For classification, an ensemble can use majority voting.

Example:

```text
Model 1 -> Class A
Model 2 -> Class B
Model 3 -> Class A

Final prediction -> Class A
```

For regression, it can average the numerical predictions.

Suppose three models predict a house price:

```text
Model 1: 30 lakh
Model 2: 32 lakh
Model 3: 34 lakh
```

Average prediction:

\[
\hat{y} = \frac{30+32+34}{3}=32
\]

The combined prediction is ₹32 lakh.

### 2.5 Why does bagging help?

Decision trees can change considerably when the training observations change. This makes them high-variance models in many settings.

Bagging reduces the influence of an individual model's unusual predictions by averaging or voting across several models.

**Important:** Bagging primarily targets variance reduction. It does not guarantee that every model will improve, and it does not necessarily remove bias.

### 2.6 Advantages and limitations

**Advantages**
- Can reduce variance and overfitting.
- Often works well with unstable models such as decision trees.
- Models can generally be trained independently, allowing parallel computation.

**Limitations**
- Requires more computation and memory than one model.
- May provide limited improvement when base models are highly correlated.
- Is less interpretable than a single small decision tree.

---

## 3. Random Forest

### 3.1 What is Random Forest?

Random Forest is an ensemble algorithm that combines many decision trees.

It uses two sources of randomness:

1. **Bootstrap sampling:** Different training samples are used to build trees.
2. **Random feature selection:** At each split, the tree considers a randomly selected subset of features rather than every feature.

These mechanisms help the trees make different predictions, making their combination more useful.

### 3.2 Random Forest architecture

```text
                 Training Dataset
                        |
          +-------------+-------------+
          |             |             |
       Sample 1      Sample 2      Sample 3
          |             |             |
       Decision       Decision       Decision
        Tree 1         Tree 2         Tree 3
          |             |             |
          +-------------+-------------+
                        |
                  Voting / Average
                        |
                  Final Prediction
```

Each decision tree is trained separately. The ensemble combines the resulting predictions.

### 3.3 Random Forest classification

Suppose the trees predict whether a tumor is benign or malignant.

```text
Tree 1 -> Benign
Tree 2 -> Malignant
Tree 3 -> Benign
Tree 4 -> Benign
Tree 5 -> Malignant
```

Votes:
- Benign: 3
- Malignant: 2

The final prediction is **Benign**, because it receives the majority of votes.

### 3.4 Random Forest regression

For regression, the predictions can be averaged.

If the trees predict:

\[
\hat{y}_1=25,\quad \hat{y}_2=28,\quad \hat{y}_3=29
\]

Then:

\[
\hat{y}=\frac{25+28+29}{3}=27.33
\]

### 3.5 Python practical

**File:** `machine_learning/05_supervised_learning/01_random_forest.py`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tree = DecisionTreeClassifier(random_state=42)

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

tree_predictions = tree.predict(X_test)
forest_predictions = forest.predict(X_test)

print("Tree accuracy:",
      accuracy_score(y_test, tree_predictions))

print("Forest accuracy:",
      accuracy_score(y_test, forest_predictions))
```

### 3.6 Code explanation

- `load_breast_cancer()` loads a built-in binary classification dataset.
- `X` contains the input features; `y` contains the target labels.
- `train_test_split()` separates data for training and evaluation.
- `test_size=0.2` reserves 20% of the data for testing.
- `stratify=y` helps preserve class proportions in both splits.
- `random_state=42` makes the split reproducible.
- `n_estimators=100` builds 100 trees.
- `fit()` trains the models.
- `predict()` produces predictions for unseen test examples.
- `accuracy_score()` calculates the fraction of correct predictions.

### 3.7 Actual practical result

```text
Decision Tree accuracy: 0.9122807017543859
Random Forest accuracy: 0.956140350877193
```

Approximately:

| Model | Accuracy |
|---|---:|
| Decision Tree | 91.23% |
| Random Forest | 95.61% |

The Random Forest performed better on this particular split. This does not prove that it will always outperform a decision tree.

### 3.8 Important Random Forest hyperparameters

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of trees |
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required in a leaf |
| `max_features` | Number or proportion of features considered at each split |
| `max_samples` | Number or proportion of samples drawn for each tree when applicable |
| `random_state` | Controls reproducibility |

Increasing the number of trees may improve stability, but also increases training and prediction cost. A deeper tree can capture more complex patterns but may overfit.

### 3.9 Feature importance

Random Forest can estimate the relative importance of each feature using reductions in impurity across the trees.

Example:

```python
importances = forest.feature_importances_

for i in importances.argsort()[::-1][:5]:
    print(data.feature_names[i], importances[i])
```

My observed top features were:

```text
worst area
worst concave points
worst radius
mean concave points
worst perimeter
```

**Interpretation:** These features received relatively high impurity-based importance in this fitted forest.

**Limitations:**
- Importance does not prove causation.
- Correlated features can share or distort importance.
- Importance does not directly measure how much test accuracy would fall if a feature were removed.

Permutation importance is an alternative: shuffle one feature's values and measure how much model performance decreases.

---

## 4. Boosting

### 4.1 What is Boosting?

Boosting builds a strong ensemble by adding models sequentially.

Instead of training every model independently, each new stage attempts to improve the existing ensemble.

```text
Training Data
     |
   Model 1
     |
Existing Ensemble's Errors / Loss
     |
   Model 2
     |
Updated Ensemble
     |
   Model 3
     |
Final Combined Model
```

The precise correction mechanism depends on the boosting algorithm.

### 4.2 Bagging versus Boosting

| Property | Bagging | Boosting |
|---|---|---|
| Training | Usually independent models | Sequential stages |
| Main idea | Combine models trained on varied samples | Improve the existing ensemble |
| Common benefit | Reduce variance | Reduce bias and/or improve predictive fit |
| Parallelism | Base models can often be trained in parallel | Stages generally depend on previous stages |
| Example | Random Forest | Gradient Boosting, XGBoost |

Boosting can also overfit, particularly with overly complex models or excessive training stages. It is not inherently immune to overfitting.

### 4.3 Main boosting algorithms

**AdaBoost**
- Gives greater weight to training examples that were previously misclassified.
- Later learners pay more attention to difficult examples.

**Gradient Boosting**
- Adds models that follow the negative gradient of a chosen loss function.
- For squared-error regression, this is closely related to fitting residuals.

**XGBoost**
- An optimized gradient-boosting implementation.
- Includes regularization and efficient training techniques.

---

## 5. Gradient Boosting

### 5.1 Basic intuition

Imagine predicting house prices. A first model makes imperfect predictions. A second model is trained to improve those predictions, and further models continue improving the ensemble.

For squared-error regression, the residual for a training example is:

\[
r_i=y_i-\hat{y}_i
\]

Where:
- \(y_i\) = actual target value.
- \(\hat{y}_i\) = current prediction.
- \(r_i\) = residual.

A positive residual means the model underpredicted. A negative residual means it overpredicted.

Gradient Boosting generalizes this idea beyond ordinary residuals by using the negative gradient of the loss function.

### 5.2 Simplified numerical example

Suppose the actual house price is ₹40 lakh.

```text
Initial prediction = 30 lakh
Actual value       = 40 lakh

Residual = 40 - 30 = +10 lakh
```

A new tree might learn patterns associated with this underprediction. Its contribution is scaled by the learning rate before being added to the current prediction.

For illustration:

```text
Current prediction: 30 lakh
Scaled correction:   4 lakh
Updated prediction: 34 lakh
```

The remaining residual is ₹6 lakh. This is a simplified explanation of the idea; real Gradient Boosting learns corrections across many training examples and uses the selected loss function.

### 5.3 Mathematical view

At boosting stage \(m\):

\[
F_m(x)=F_{m-1}(x)+\eta h_m(x)
\]

Where:
- \(F_{m-1}(x)\) = previous ensemble prediction.
- \(h_m(x)\) = new tree's learned contribution.
- \(\eta\) = learning rate.
- \(F_m(x)\) = updated prediction.

A small learning rate limits each tree's contribution.

### 5.4 Gradient Boosting regression practical

**File:** `machine_learning/05_supervised_learning/02_gradient_boosting.py`

```python
from sklearn.ensemble import GradientBoostingRegressor

X = [[500], [700], [900], [1100], [1300], [1500]]
y = [20, 25, 32, 38, 45, 52]

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=1,
    random_state=42
)

model.fit(X, y)

prediction = model.predict([[1000]])

print("Predicted price:", prediction[0], "lakh")
```

Output from my practice:

```text
Predicted price: 32.078745773687096 lakh
```

These are invented example data, not actual real-estate data. The prediction demonstrates how the model runs; it is not a reliable market valuation.

### 5.5 Parameters explained

- `n_estimators=50`: Build 50 boosting stages.
- `learning_rate=0.1`: Scale each new tree's contribution.
- `max_depth=1`: Restrict each tree to a shallow structure.
- `random_state=42`: Support reproducibility.

The number of stages and learning rate interact. A smaller learning rate often needs more stages, so tune them together rather than assuming one fixed setting is best.

### 5.6 Advantages and limitations

**Advantages**
- Can model nonlinear relationships.
- Often performs well on structured, tabular data.
- Builds a flexible model by adding small corrections.

**Limitations**
- Sequential training can be slower than parallel bagging.
- Requires careful tuning.
- Can overfit when model complexity or the number of stages is excessive.
- Feature importance should not be interpreted as causal evidence.

---

## 6. XGBoost

### 6.1 What is XGBoost?

XGBoost stands for **Extreme Gradient Boosting**.

It is a widely used implementation of gradient-boosted decision trees, designed with features such as regularization and efficient computation.

It is commonly useful for structured data containing rows and columns, such as customer, financial, business, or medical measurements.

### 6.2 How XGBoost works

```text
                 Training Data
                      |
             Initial Prediction
                      |
               Calculate Loss
                      |
            Train a New Decision Tree
                      |
          Add a Scaled Tree Contribution
                      |
              Updated Prediction
                      |
           Repeat for More Trees
                      |
                Final Ensemble
```

Each stage improves the existing prediction according to the chosen objective and optimization procedure.

### 6.3 Objective function

A simplified XGBoost objective can be written as:

\[
\text{Objective}=\text{Training Loss}+\text{Regularization}
\]

- **Training loss** measures how poorly the predictions fit the training targets.
- **Regularization** penalizes model complexity to help discourage overfitting.

XGBoost uses a more specific objective involving tree structure and leaf weights; the formula above expresses the central idea rather than the complete implementation.

### 6.4 XGBoost classification practical

**File:** `machine_learning/05_supervised_learning/03_xgboost.py`

Install the package if it is not already installed:

```bash
python -m pip install xgboost
```

Code:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("XGBoost accuracy:", accuracy)
```

### 6.5 XGBoost parameter explanations

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of boosting trees |
| `learning_rate` | Scales the contribution of each tree |
| `max_depth` | Limits tree depth |
| `eval_metric` | Selects the evaluation metric used for monitoring |
| `random_state` | Supports reproducibility |

Other useful XGBoost parameters include:

- `subsample`: Fraction of training observations sampled for each boosting iteration.
- `colsample_bytree`: Fraction of features sampled for each tree.
- `reg_lambda`: L2 regularization strength on leaf weights.
- `reg_alpha`: L1 regularization strength on leaf weights.
- `min_child_weight`: Controls the minimum sum of instance weights needed in a child; it is not simply a count of rows.

Do not tune every parameter at once. Start with a reasonable baseline, measure performance, and change a few parameters systematically.

### 6.6 My model comparison

All three models were tested on the same breast-cancer dataset with the same train/test split.

| Model | Accuracy | Approximate correct predictions out of 114 |
|---|---:|---:|
| Decision Tree | 91.23% | 104 |
| Random Forest | 95.61% | 109 |
| XGBoost | 94.74% | 108 |

The XGBoost result was:

```text
XGBoost accuracy: 0.9473684210526315
```

The model correctly classified approximately 94.74% of the test examples.

**Conclusion:** Random Forest scored slightly higher on this particular split. The difference is only about one test example, so this result is not enough to establish that Random Forest is generally superior.

### 6.7 XGBoost limitations

- More parameters can make tuning harder.
- Training and inference can require substantial resources on large datasets.
- Models are less interpretable than simple decision trees.
- Performance depends on the data, hyperparameters, and validation procedure.
- Accuracy alone can be misleading when classes are imbalanced or different errors have different costs.

---

## 7. Stacking

Stacking is another ensemble technique worth knowing even though our practical work focused on bagging and boosting.

It trains a **meta-model** to combine predictions from different base models.

```text
                 Training Data
                      |
          +-----------+-----------+
          |           |           |
       Model A     Model B      Model C
          |           |           |
       Predict A   Predict B   Predict C
          |           |           |
          +-----------+-----------+
                      |
                 Meta-model
                      |
               Final Prediction
```

For example, the base models might be Logistic Regression, Random Forest, and Gradient Boosting. The meta-model learns how to combine their outputs.

A critical implementation detail: the meta-model should generally be trained using out-of-fold predictions from the base models, rather than predictions they made on the same data used to fit them. This helps prevent data leakage and overly optimistic training performance.

---

## 8. Model Evaluation and Selection

### 8.1 Accuracy

For classification:

\[
\text{Accuracy}=
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

Plain text:

`Accuracy = correct predictions / total predictions`

Accuracy is useful when classes and error costs make it appropriate, but it is not sufficient for every problem.

### 8.2 Other metrics

Use the metric that matches the real task:

- **Precision:** Of predicted positives, how many were actually positive?
- **Recall:** Of actual positives, how many did the model find?
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures ranking discrimination across classification thresholds.
- **MAE / RMSE:** Common regression error metrics.

You studied classification metrics earlier, so use those concepts when comparing classifiers rather than relying on accuracy alone.

### 8.3 Cross-validation

Cross-validation evaluates a model across several train/validation splits.

```text
Dataset
   |
   +-- Fold 1: Validate on part 1
   +-- Fold 2: Validate on part 2
   +-- Fold 3: Validate on part 3
   +-- Fold 4: Validate on part 4
   +-- Fold 5: Validate on part 5
                 |
          Compare Results
```

In k-fold cross-validation, the data is divided into \(k\) folds. Each fold takes a turn as the validation set while the others are used for training.

The average score gives a more informative estimate than one split, though variation across folds also matters. Keep a final test set separate when using cross-validation for model selection.

### 8.4 Choosing an ensemble

Consider:
1. Performance on suitable validation data.
2. Generalization across splits.
3. Training and prediction speed.
4. Memory requirements.
5. Interpretability requirements.
6. The cost of different prediction errors.
7. Operational complexity.

The model with the highest single accuracy score is not automatically the best model for deployment.

---

## 9. Common Mistakes

1. Assuming an ensemble must outperform every individual model.
2. Assuming XGBoost is always better than Random Forest.
3. Evaluating only on training data.
4. Tuning hyperparameters against the final test set.
5. Using accuracy alone for imbalanced classification.
6. Treating feature importance as proof of causation.
7. Ignoring data leakage during preprocessing or stacking.
8. Increasing the number of trees without checking whether performance improves.
9. Comparing results from different train/test splits as if the comparison were controlled.
10. Assuming one successful experiment proves generalization.

---

## 10. Quick Revision Table

| Algorithm | Main idea | Training style | Common use |
|---|---|---|---|
| Bagging | Combine models trained on bootstrap samples | Parallel | Variance reduction |
| Random Forest | Bagging + random feature selection | Parallel trees | Classification and regression |
| AdaBoost | Increase focus on misclassified observations | Sequential | Classification and regression |
| Gradient Boosting | Add models following the loss gradient | Sequential | Classification and regression |
| XGBoost | Optimized, regularized gradient boosting | Sequential boosting stages | Structured/tabular data |
| Stacking | Learn how to combine model predictions | Base models + meta-model | Combining diverse models |

---

## 11. Practice Questions

### Conceptual questions

1. What is ensemble learning, and why is it useful?
2. Explain bootstrap sampling with an example.
3. Why does Random Forest use random feature selection?
4. Differentiate bagging and boosting.
5. What is a residual?
6. Explain the role of learning rate in Gradient Boosting.
7. What does regularization do in XGBoost?
8. Why can feature importance be misleading when features are correlated?
9. Why should a final test set not be repeatedly used for hyperparameter tuning?
10. Explain how stacking differs from Random Forest.

### Practical tasks

1. Compare a decision tree and Random Forest using the same split.
2. Change `n_estimators` from 100 to 200 and observe the result.
3. Change `max_depth` and compare performance.
4. Compare Gradient Boosting and XGBoost on the same dataset.
5. Calculate accuracy and a confusion matrix for a classifier.
6. Use cross-validation to compare models more reliably.
7. Inspect feature importance and explain its limitations.

## 12. Final Summary

Ensemble learning combines multiple models to improve prediction quality or robustness.

- Bagging trains models independently on bootstrap samples and aggregates their predictions.
- Random Forest combines bagging with random feature selection.
- Boosting builds an ensemble sequentially, improving the model according to an algorithm-specific learning objective.
- Gradient Boosting uses the negative gradient of a loss function; residuals provide an intuitive explanation for squared-error regression.
- XGBoost extends gradient boosting with regularization and efficient computation.
- Stacking uses a meta-model to learn how to combine predictions from other models.

**Core principle:** Evaluate models with a reliable validation procedure, choose appropriate metrics, and never assume a more complex model is automatically better.
