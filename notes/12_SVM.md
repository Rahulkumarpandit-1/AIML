# Support Vector Machine (SVM)

**Roadmap status:** Practiced  
**Next step:** Consolidate notes and verify local code execution.

## 1. What is SVM?

Support Vector Machine (SVM) is a supervised machine learning algorithm mainly used for classification. It can also be used for regression through Support Vector Regression (SVR).

SVM tries to find a decision boundary that separates classes while maximizing the margin between the boundary and the nearest training points.

### Important terms

- **Decision boundary:** The line or surface separating predicted classes.
- **Margin:** The distance between the decision boundary and its nearest training points.
- **Support vectors:** Training points that determine the position of the boundary and margin.
- **Hyperplane:** A general name for a decision boundary. It is a line in 2D and a plane in 3D.

**SVM vs KNN**

- KNN predicts using nearby training examples.
- SVM learns a decision boundary during training.

## 2. Decision boundary mathematics

A linear SVM decision boundary is represented by:

**Mathematical form:**

\(w^T x+b=0\)

**Plain-text form:**
`w1*x1 + w2*x2 + ... + b = 0`

Where:

- \(x\): Input features.
- \(w\): Learned weights that determine the boundary's orientation.
- \(b\): Bias that shifts the boundary.
- \(w^T x\): Dot product of weights and features.

### Example

Consider this boundary:

`2*x1 + x2 - 6 = 0`

Rearranging:

`x2 = 6 - 2*x1`

This is a straight line in two dimensions.

For a new point, SVM calculates a decision score:

`f(x) = w.T @ x + b`

For a binary classifier using the usual sign convention:

- Positive score → one class.
- Negative score → the other class.
- Zero → the decision boundary.

The exact class labels depend on how the model encodes the classes.

## 3. Support vectors and margin

Support vectors are the training points closest to the decision boundary. They are especially important because they determine the learned margin and boundary.

For a canonical linear SVM, the margin boundaries are:

- `w.T @ x + b = +1`
- `w.T @ x + b = -1`

The decision boundary lies between them:

`w.T @ x + b = 0`

The total margin width is:

\[
\text{Margin width}=\frac{2}{\|w\|}
\]

Here, \(\|w\|\) is the Euclidean length of the weight vector:

\[
\|w\|=\sqrt{w_1^2+w_2^2+\cdots+w_n^2}
\]

### Numerical example

Suppose:

`w = (2, 1)`

Then:

\[
\|w\|=\sqrt{2^2+1^2}=\sqrt{5}\approx2.236
\]

Therefore:

\[
\text{Margin width}=\frac{2}{\sqrt{5}}\approx0.894
\]

A smaller weight-vector norm corresponds to a wider margin in this formulation.

**Important:** These margin equations describe the canonical linear SVM formulation. For nonlinear kernels, the learned boundary is not generally a straight line in the original feature space.

## 4. Hard margin vs soft margin

### Hard-margin SVM

- Requires the training data to be perfectly linearly separable.
- Does not allow classification or margin violations.
- Can be highly sensitive to outliers.
- Is unsuitable for many real-world datasets.

### Soft-margin SVM

- Allows some points to violate the margin.
- Can tolerate overlapping classes and noisy data.
- Uses the parameter `C` to control the penalty for violations.

### Understanding C

| C value | General effect |
|---|---|
| Small C | Allows more violations; often favors a wider, more regularized margin |
| Large C | Penalizes violations more heavily; often fits training data more strictly |

A large `C` does not guarantee better test accuracy. It may cause overfitting.

## 5. What is a kernel?

A kernel allows SVM to learn nonlinear decision boundaries by comparing data points in an implicit feature space, without necessarily constructing every transformed feature explicitly.

Use a linear boundary when the relationship is approximately linear. Consider nonlinear kernels when a straight boundary cannot separate the classes well.

### Linear kernel

`K(x, z) = x.T @ z`

- Useful when the boundary is approximately linear.
- Often effective for high-dimensional, sparse data such as text features.

### Polynomial kernel

`K(x, z) = (gamma * (x.T @ z) + r)^d`

Where:

- `gamma`: Controls the scaling of the dot product.
- `r`: Constant term, commonly called `coef0`.
- `d`: Polynomial degree.

It can learn nonlinear boundaries involving polynomial relationships.

### RBF kernel

`K(x, z) = exp(-gamma * ||x - z||^2)`

The kernel similarity is high for nearby points and decreases as the distance increases.

**Understanding gamma:**

- Low gamma → broader influence of points and generally smoother boundaries.
- High gamma → more local influence and potentially more complex boundaries.

Very high gamma, especially combined with high `C`, can lead to overfitting.

## 6. SVM implementation using scikit-learn

**Code file:** `machine_learning/08_SVM/01_svm_kernels.py`

This example compares linear SVM with RBF SVM using a nonlinear dataset.

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a nonlinear binary classification dataset
X, y = make_moons(
    n_samples=500,
    noise=0.25,
    random_state=42
)

# Split before fitting preprocessing or models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Linear SVM": SVC(kernel="linear", C=1.0),
    "RBF SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )
}

for name, svm in models.items():
    model = make_pipeline(
        StandardScaler(),
        svm
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.3f}")
```

### How the code works

1. `make_moons()` creates two curved classes.
2. `train_test_split()` reserves unseen examples for evaluation.
3. `StandardScaler()` standardizes the features.
4. `make_pipeline()` combines scaling and SVM into one workflow.
5. `SVC(kernel="linear")` learns a linear boundary.
6. `SVC(kernel="rbf")` can learn a nonlinear boundary.
7. `fit()` trains the model using the training data.
8. `predict()` predicts the test labels.
9. `accuracy_score()` calculates the fraction of test predictions that are correct.

The RBF model may outperform the linear model on this dataset, but the actual result depends on the data and parameters. Do not assume a kernel is always better without evaluating it.

## 7. Why use StandardScaler in a pipeline?

SVM can be sensitive to feature scales because distances and dot products affect its calculations.

Imagine a dataset with:

- Age: values from 18 to 60.
- Annual income: values from 20,000 to 2,000,000.

Without scaling, income can dominate calculations because its numerical values are much larger.

`StandardScaler` standardizes each feature using its mean and standard deviation:

`z = (x - mean) / standard_deviation`

A pipeline ensures the scaler is fitted on training data during `fit()` and then applied to test data using the training statistics.

This helps prevent data leakage. Never fit the scaler on the entire dataset before splitting it into training and test sets.

## 8. Evaluation and hyperparameter tuning

Accuracy alone may be insufficient, especially when classes are imbalanced.

Depending on the problem, also examine:

- Precision
- Recall
- F1-score
- Confusion matrix

Important SVM parameters include:

- `kernel`: `"linear"`, `"poly"`, or `"rbf"`.
- `C`: Penalty for margin violations.
- `gamma`: Kernel influence parameter, especially for RBF and polynomial kernels.
- `degree`: Polynomial degree for the polynomial kernel.

Use cross-validation on the training data to compare parameter combinations. Keep the test set for final evaluation rather than repeatedly tuning against it.

## 9. Advantages and limitations

### Advantages

- Can learn linear and nonlinear decision boundaries.
- Maximizing the margin can support good generalization.
- Often performs well on high-dimensional datasets.
- Uses support vectors to define the decision boundary.

### Limitations

- Feature scaling is often important.
- Parameter and kernel selection can require experimentation.
- Training can become expensive on large datasets.
- Performance can suffer with noisy, overlapping classes.
- Probability estimates require additional calibration when needed.
- SVM is not automatically the best model for every dataset.

## 10. Common mistakes

1. Forgetting to scale features.
2. Fitting preprocessing on the entire dataset before splitting.
3. Assuming high `C` always improves generalization.
4. Assuming high `gamma` always improves performance.
5. Choosing a complicated kernel without comparing a simpler baseline.
6. Tuning hyperparameters using the test set.
7. Treating high training accuracy as proof of good generalization.

## 11. Quick revision sheet

- **SVM:** Finds a decision boundary with a large margin.
- **Hyperplane:** The decision boundary.
- **Support vectors:** Training points that determine the margin and boundary.
- **Margin width:** `2 / ||w||` for the canonical linear formulation.
- **Hard margin:** No margin violations allowed.
- **Soft margin:** Allows violations.
- **C:** Controls how strongly violations are penalized.
- **Linear kernel:** For approximately linear boundaries.
- **Polynomial kernel:** Can model polynomial nonlinear relationships.
- **RBF kernel:** Useful for many nonlinear boundaries.
- **High gamma:** More local influence; potentially more complex boundary.
- **StandardScaler:** Helps put features on comparable scales.
- **Pipeline:** Keeps preprocessing and model training together and helps avoid leakage.
- **Cross-validation:** Helps select parameters without repeatedly using the test set.

## 12. Self-test

1. What is the difference between a support vector and an ordinary training point?
2. If `||w|| = 4`, what is the margin width?
3. What is the general effect of increasing `C`?
4. Why might you choose RBF instead of a linear kernel?
5. What is the general effect of increasing RBF `gamma`?
6. Why should the scaler be fitted only on the training data?
7. Why should you avoid repeatedly tuning against the test set?

**Progress rule:** Keep SVM marked as **Practiced** until you have reviewed these notes and verified the implementation locally. Move to **Mastered** only after you can explain the concepts and solve practical problems independently.
