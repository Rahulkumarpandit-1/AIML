# Decision Trees — Complete Notes

**Roadmap section:** Supervised Machine Learning  
**Status:** Learning  
**Goal:** Understand how Decision Trees make predictions, how they choose splits, and how to control overfitting.

---

## 1. What is a Decision Tree?

A Decision Tree is a supervised machine learning algorithm that makes predictions by asking a sequence of questions about input features.

It resembles a flowchart: each question directs an example down a branch until it reaches a final prediction.

### Example

Suppose we want to predict whether a student passes an exam based on study hours.

| Study hours | Result |
|---:|---|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Fail |
| 5 | Pass |
| 6 | Pass |
| 7 | Pass |
| 8 | Pass |

A simplified tree might learn the rule:

- If study hours ≤ 4.5, predict Fail.
- If study hours > 4.5, predict Pass.

The value 4.5 is a decision threshold separating the two groups in this example.

### Important terminology

- **Root node:** The first node where the tree begins.
- **Decision node:** A node that asks a question about a feature.
- **Branch:** The path followed after a decision.
- **Leaf node:** A final node that produces a prediction.
- **Feature:** An input variable, such as study hours.
- **Target:** The value we want to predict, such as Pass or Fail.
- **Split:** Dividing examples into groups using a feature and a threshold.

### Types of Decision Trees

**DecisionTreeClassifier**
- Predicts categories or classes.
- Examples: Pass/Fail, Spam/Not Spam, Disease/No Disease.

**DecisionTreeRegressor**
- Predicts numerical values.
- Examples: House prices, salaries, temperatures.

---

## 2. How does a Decision Tree learn?

The tree searches for feature-based questions that divide the training data into more useful groups.

For classification, a useful split generally separates different classes into purer groups.

Suppose a parent node contains:

- 5 Pass examples
- 5 Fail examples

The node has a mixture of both classes.

A good split might create two children:

- Left child: 5 Fail examples.
- Right child: 5 Pass examples.

Both children are pure, so the split perfectly separates these training examples.

A poor split might leave both children containing similar mixtures of Pass and Fail.

**Main principle:** For classification, a good split reduces class impurity.

The algorithm repeats this process recursively, creating more decision nodes until a stopping condition is reached.

---

## 3. Gini impurity

Gini impurity measures how mixed the classes are within a node.

### Formula

Gini = 1 − Σ(pᵢ²)

Here:
- pᵢ is the proportion of examples belonging to class i.
- Σ means to add the squared proportions for all classes.
- A node containing only one class has Gini impurity 0.

### Example 1: Pure node

Suppose a node contains 10 Pass examples and no Fail examples.

Pass proportion = 10/10 = 1

Fail proportion = 0/10 = 0

Gini = 1 − (1² + 0²)

Gini = 0

The node is pure.

### Example 2: Mixed node

Suppose a node contains 6 Pass examples and 4 Fail examples.

Pass proportion = 6/10 = 0.6

Fail proportion = 4/10 = 0.4

Gini = 1 − (0.6² + 0.4²)

Gini = 1 − (0.36 + 0.16)

**Gini = 0.48**

### Interpretation

- Gini = 0: Pure node.
- For binary classification, Gini = 0.5: Maximum impurity, when the two classes are equally represented.
- Lower Gini means a purer node.

### Weighted Gini after a split

We must consider both children and their sizes.

Weighted Gini = (nL/n) × GiniL + (nR/n) × GiniR

Where:
- n = total number of examples in the parent.
- nL and nR = numbers of examples in the left and right children.
- GiniL and GiniR = impurity of the respective children.

Example:

- Left child: 5 examples, Gini = 0.
- Right child: 5 examples, Gini = 0.
- Parent: 10 examples.

Weighted Gini = (5/10 × 0) + (5/10 × 0) = 0.

The split completely separates the classes.

**Remember:** The best split generally has the lowest weighted child impurity among the candidate splits.

---

## 4. Entropy and Information Gain

Entropy is another measure of class impurity.

### Entropy formula

Entropy = −Σ(pᵢ × log₂(pᵢ))

Here:
- pᵢ = proportion of class i.
- log₂ = logarithm with base 2.
- A term for pᵢ = 0 is treated as zero.

### Examples

Pure node:

- 10 Pass, 0 Fail.
- Entropy = 0.

Balanced node:

- 5 Pass, 5 Fail.
- Entropy = 1 bit.

Mixed node:

- 7 Pass, 3 Fail.
- Entropy ≈ 0.881 bits.

For binary classification, entropy is highest when both classes have equal proportions.

### Information Gain

Information Gain measures how much entropy decreases after a split.

Information Gain = Parent Entropy − Weighted Child Entropy

A larger information gain indicates a greater reduction in impurity.

Example:

- Parent entropy = 1.
- Weighted child entropy = 0.

Information Gain = 1 − 0 = 1 bit.

This is a perfect split for those examples.

### Gini vs Entropy

| Property | Gini | Entropy |
|---|---|---|
| Measures | Class impurity | Class impurity |
| Pure node | 0 | 0 |
| Balanced binary node | 0.5 | 1 bit |
| Better split | Lower weighted Gini | Higher information gain |
| Scikit-learn option | `criterion="gini"` | `criterion="entropy"` |

Both are valid splitting criteria. You do not need to calculate every split manually when using scikit-learn.

---

## 5. How Decision Trees grow

A Decision Tree repeatedly:

1. Examines possible feature splits.
2. Measures the quality of those splits.
3. Selects a suitable split.
4. Creates child nodes.
5. Repeats the process on the children.
6. Stops when a stopping condition is met.

### Important hyperparameters

A hyperparameter is a setting chosen before or during model selection that controls how the algorithm learns.

| Parameter | Purpose |
|---|---|
| `max_depth` | Maximum number of decision levels |
| `min_samples_split` | Minimum number of examples required to split a node |
| `min_samples_leaf` | Minimum number of examples allowed in a leaf |
| `max_leaf_nodes` | Maximum number of leaf nodes |
| `criterion` | Impurity measure used to choose classification splits |
| `ccp_alpha` | Complexity penalty used for cost-complexity pruning |

Example:

`DecisionTreeClassifier(max_depth=3)`

This restricts the tree to a maximum depth of 3.

Increasing restrictions generally produces a simpler tree, but excessive restrictions can cause underfitting.

---

## 6. Overfitting and Underfitting

### Overfitting

Overfitting occurs when a model learns the training examples too closely, including patterns that do not generalize well to unseen examples.

Typical signs:
- Very high training accuracy.
- Noticeably lower validation or test accuracy.
- A tree that is unnecessarily complex.

Example:

Training accuracy = 100%

Validation accuracy = 85%

This gap may indicate overfitting, although it is not proof by itself.

### Underfitting

Underfitting occurs when the model is too simple to learn important patterns.

Typical signs:
- Low training accuracy.
- Low validation accuracy.

### How to control overfitting

- Restrict `max_depth`.
- Increase `min_samples_split`.
- Increase `min_samples_leaf`.
- Restrict `max_leaf_nodes`.
- Use cost-complexity pruning with `ccp_alpha`.
- Select settings using validation data or cross-validation.

---

## 7. Pre-pruning and Post-pruning

Pruning reduces unnecessary tree complexity.

### Pre-pruning

Pre-pruning stops the tree from growing too much.

Examples:
- `max_depth=3`
- `min_samples_leaf=5`
- `min_samples_split=10`

These restrictions are applied while the tree is being built.

### Post-pruning

Post-pruning simplifies a tree after it has grown.

Scikit-learn supports cost-complexity pruning using `ccp_alpha`.

The simplified cost-complexity objective is:

Rα(T) = R(T) + α × |T leaves|

Where:
- T = the tree.
- R(T) = total weighted leaf impurity under the chosen tree criterion.
- α = the complexity penalty.
- |T leaves| = number of leaf nodes.
- Rα(T) = combined impurity and complexity objective.

The algorithm balances predictive purity against tree size.

### Understanding `ccp_alpha`

- `ccp_alpha=0.0`: No complexity penalty.
- A larger `ccp_alpha`: Generally encourages a smaller tree.
- Too much pruning: May cause underfitting.
- Pruning does not guarantee better test accuracy.

A tree's size can remain unchanged across multiple alpha values because some penalty changes do not remove any branches.

### Interpreting our experiment

| Alpha | Depth | Leaves | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|---:|
| 0.000 | 6 | 15 | 100% | 95.6% |
| 0.001 | 6 | 15 | 100% | 95.6% |
| 0.005 | 5 | 11 | 99.4% | 94.7% |
| 0.010 | 2 | 4 | 96.8% | 94.7% |
| 0.020 | 2 | 4 | 96.8% | 94.7% |

The largest validation accuracy was 95.6%, achieved by both 0.000 and 0.001.

Because the code selected the first maximum, it selected `alpha=0.0`.

**Conclusion:** Pruning simplified the tree at higher alpha values, but it did not improve validation accuracy in this experiment.

---

## 8. Training, Validation and Test Sets

These datasets have different purposes.

### Training set
Used to fit the model and learn patterns.

### Validation set
Used to compare models and choose settings, such as `max_depth` or `ccp_alpha`.

### Test set
Used for a final performance evaluation after model selection.

Example split:

- 60% training.
- 20% validation.
- 20% testing.

These percentages are illustrative, not mandatory.

A common procedure is:
1. Split off the test set.
2. Split the remaining data into training and validation sets.
3. Train candidate models using training data.
4. Select settings using validation performance.
5. Evaluate the selected model on the test set.

**Important:** Do not repeatedly tune a model based on test results. That leaks information from the test set into model selection.

Cross-validation is another way to estimate validation performance more robustly.

---

## 9. Feature Importance

A Decision Tree can estimate how much each feature contributed to reducing impurity across its splits.

In scikit-learn:

`model.feature_importances_`

The importance values are nonnegative and sum to 1 for a fitted tree with features.

Example from our experiment:

| Feature | Importance |
|---|---:|
| worst concave points | 0.7249 |
| worst area | 0.1113 |
| worst perimeter | 0.0459 |
| worst texture | 0.0274 |
| area error | 0.0271 |

The feature `worst concave points` contributed the most to impurity reductions in this fitted model.

An importance of 0.7249 means approximately 72.49% of the total impurity-reduction importance was attributed to that feature.

### Limitations

- Feature importance does not establish causation.
- It does not indicate model accuracy.
- Impurity-based importance can favor features with many possible split points.
- Correlated features can divide or distort importance.
- Importance can change across datasets or fitted models.

Use feature importance as a way to inspect the fitted model, not as proof of a real-world causal relationship.

---

## 10. Building a Decision Tree with scikit-learn

### Code file

`machine_learning/03_decision_tree/01_decision_tree_basics.py`

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Input feature: study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Target: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### Code explanation

- `X` stores input features.
- `y` stores the correct labels.
- `train_test_split()` separates training and testing examples.
- `test_size=0.25` reserves 25% of the examples for testing.
- `random_state=42` makes the random split reproducible.
- `stratify=y` attempts to preserve class proportions in both sets.
- `DecisionTreeClassifier()` creates the model.
- `model.fit()` trains it.
- `model.predict()` produces predictions.
- `accuracy_score()` measures the fraction of correct predictions.

This is a tiny teaching example. Real datasets need enough examples for reliable training and evaluation.

---

## 11. Comparing tree depths

### Code file

`machine_learning/03_decision_tree/02_tree_depth.py`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

depths = [1, 2, 3, 5, 10, None]

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(
        f"Depth={depth} | "
        f"Train={train_accuracy:.3f} | "
        f"Test={test_accuracy:.3f}"
    )
```

### How to interpret it

- A shallow tree may miss important patterns.
- A deeper tree can learn more detailed patterns.
- A very deep tree may fit the training data extremely well but generalize poorly.

This example is for learning the effect of depth. For rigorous model selection, use a validation set or cross-validation rather than choosing a depth from repeated test scores.

---

## 12. Decision Tree evaluation

Use evaluation metrics appropriate for the classification task.

- **Accuracy:** Fraction of all predictions that are correct.
- **Precision:** Of the examples predicted positive, the fraction that are actually positive.
- **Recall:** Of the actual positive examples, the fraction correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts of true negatives, false positives, false negatives and true positives.

These metrics are not unique to Decision Trees; they apply to classification models generally.

### Important formulas

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Where:
- TP = true positives.
- TN = true negatives.
- FP = false positives.
- FN = false negatives.

Precision and recall are undefined in their basic mathematical form when their denominators are zero. Scikit-learn's metric functions provide configurable handling for such cases.

---

## 13. Advantages of Decision Trees

1. Easy to understand and explain.
2. Can model nonlinear relationships.
3. Can capture feature interactions.
4. Usually do not require feature scaling.
5. Can be used for classification and regression.
6. Can provide feature importance estimates.
7. Can work with numerical and appropriately encoded categorical inputs.

## 14. Limitations of Decision Trees

1. Deep trees can overfit.
2. Small data changes can produce different trees.
3. Greedy splitting does not guarantee the globally optimal tree.
4. Individual trees can generalize less reliably than some ensemble methods.
5. Impurity-based feature importance can be misleading.
6. Predictions outside the observed feature ranges may not extrapolate well, especially for regression trees.

---

## 15. Common mistakes

- Assuming higher depth always improves generalization.
- Assuming pruning always improves accuracy.
- Choosing hyperparameters based on test accuracy.
- Confusing entropy with information gain.
- Choosing the split with the highest weighted Gini instead of the lowest.
- Treating feature importance as causal proof.
- Reporting training accuracy as evidence of performance on unseen data.
- Assuming every tree becomes smaller for every increase in `ccp_alpha`.
- Forgetting that `DecisionTreeClassifier` predicts classes while `DecisionTreeRegressor` predicts numerical values.

---

## 16. Quick revision sheet

| Concept | Remember |
|---|---|
| Decision Tree | Predicts through a sequence of feature-based decisions |
| Root | First node |
| Leaf | Final prediction node |
| Gini impurity | Lower is better for a candidate split |
| Entropy | Lower means less class uncertainty |
| Information Gain | Parent entropy minus weighted child entropy; higher is better |
| Overfitting | Model fits training patterns too closely |
| Pre-pruning | Stop growth early |
| Post-pruning | Remove unnecessary branches after growth |
| `max_depth` | Limits depth |
| `min_samples_leaf` | Sets minimum examples per leaf |
| `ccp_alpha` | Controls the complexity penalty |
| Validation set | Selects model settings |
| Test set | Final evaluation |
| Feature importance | Measures contributions to impurity reduction |
| `DecisionTreeClassifier` | Classification |
| `DecisionTreeRegressor` | Regression |

---

## 17. Self-test

Answer these without looking at the notes.

1. What is the difference between a decision node and a leaf node?
2. What does a Gini impurity of zero mean?
3. How does information gain help select a split?
4. What is the difference between pre-pruning and post-pruning?
5. What can happen when `max_depth` is too large?
6. Why should we use validation data to choose `ccp_alpha`?
7. Why does a high feature-importance score not prove causation?
8. What is the difference between a classifier and a regressor?
9. Why should the test set remain separate during model selection?
10. Does pruning guarantee higher test accuracy? Explain.

### Practical completion checklist

- [ ] Explain the Decision Tree structure without notes.
- [ ] Calculate Gini impurity for a small example.
- [ ] Explain entropy and information gain.
- [ ] Train and interpret a simple classifier.
- [ ] Compare shallow and deep trees.
- [ ] Explain pre-pruning and post-pruning.
- [ ] Interpret feature importance carefully.
- [ ] Explain the training/validation/test workflow.
- [ ] Answer the self-test.
- [ ] Complete one independent Decision Tree classification exercise.

**Current status: Learning.** Move to Practiced after completing the practical exercise, reviewing the notes, and demonstrating understanding of the main concepts.
