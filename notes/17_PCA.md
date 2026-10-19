# Principal Component Analysis (PCA)

## 1. What Is PCA?

**Principal Component Analysis (PCA)** is an unsupervised dimensionality-reduction technique that transforms original numerical features into a smaller set of new features called principal components.

Its goal is to preserve as much variance in the original data as possible while reducing the number of dimensions.

### Why use PCA?

- Reduce the number of features.
- Visualize high-dimensional datasets in 2D or 3D.
- Reduce redundancy among correlated features.
- Potentially reduce computational cost for downstream models.
- Sometimes reduce noise by removing low-variance directions.

**Important:** PCA creates new features from combinations of the original features. It does not simply select existing columns.

## 2. Variance and Principal Components

Variance measures how spread out the data is.

PCA identifies directions in which the observations have the greatest variance.

- **PC1:** captures the maximum possible variance along one direction.
- **PC2:** captures the maximum remaining variance while being perpendicular to PC1.
- **Further components:** capture as much remaining variance as possible while being perpendicular to the previous components.

The components are ordered by the variance they explain.

High variance does not always mean high predictive importance because PCA does not use target labels when selecting components.

## 3. How PCA Works

1. Prepare the numerical features and handle missing values.
2. Split the dataset into training and testing sets when evaluating a model.
3. Standardize the features when their scales differ.
4. Center the training data by subtracting each feature's mean.
5. Find principal component directions using a method such as singular value decomposition (SVD).
6. Choose how many components to retain.
7. Project the data onto those component directions.

Scikit-learn's `PCA` automatically centers its input, but it does not automatically standardize features to unit variance.

## 4. Mathematical Intuition

### A. Standardization

Formula:

z = (x − μ) / σ

Plain text: standardized value = (original value − feature mean) / feature standard deviation.

Where:
- x = original value
- μ = feature mean
- σ = feature standard deviation
- z = standardized value

Standardization is useful when numerical features have different scales. It is not always required for every dataset.

### B. Covariance Matrix

The covariance matrix describes the variance of each feature and how pairs of features vary together.

For centered data X with n observations:

Σ = XᵀX / (n − 1)

Here, Xᵀ is the transpose of X, and Σ is the sample covariance matrix.

### C. Principal Components

PCA finds eigenvectors and eigenvalues of the covariance matrix, or computes equivalent directions using SVD.

- Eigenvector: direction of a principal component.
- Eigenvalue: variance captured along that direction.

The first principal component corresponds to the largest eigenvalue.

### D. Explained Variance Ratio

EVRᵢ = λᵢ / Σⱼ λⱼ

Plain text: explained variance ratio = component eigenvalue / sum of all component eigenvalues.

The explained variance ratio measures the fraction of total variance captured by a component.

Example:
- PC1: 70%
- PC2: 20%
- PC3: 7%
- PC4: 3%

The first two components retain 90% of the total variance.

Cumulative explained variance is the sum of the explained variance ratios up to a chosen component.

## 5. Small Numerical Example

Consider this dataset:

| Observation | X1 | X2 |
|---|---:|---:|
| A | 2 | 1 |
| B | 4 | 2 |
| C | 6 | 3 |

Since X2 = X1 / 2, both features contain exactly the same underlying variation.

Their means are:
- Mean of X1 = 4
- Mean of X2 = 2

After centering:

| Observation | Centered X1 | Centered X2 |
|---|---:|---:|
| A | -2 | -1 |
| B | 0 | 0 |
| C | 2 | 1 |

The sample covariance matrix is:

Σ = [[4, 2], [2, 1]]

Its eigenvalues are 5 and 0.

Therefore:
- PC1 captures 100% of the variance.
- PC2 captures 0% of the variance.

Projecting the observations onto PC1 produces one value per observation. The shape changes from (3, 2) to (3, 1).

This is an idealized example. Real datasets usually have some variance in multiple directions.

## 6. Scikit-Learn Implementation

File: `machine_learning/06_unsupervised_learning/04_pca.py`

### A. Load and split the dataset

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Original feature count:", X.shape[1])
```

The dataset has 30 numerical features.

### B. Standardize the data

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Fit the scaler on training data only. Apply the learned transformation to test data.

### C. Reduce to two components

```python
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Before PCA:", X_train_scaled.shape)
print("After PCA:", X_train_pca.shape)

print("Explained variance ratio:",
      pca.explained_variance_ratio_)

print("Total variance retained:",
      pca.explained_variance_ratio_.sum())
```

`n_components=2` keeps two principal components.

- `fit_transform()` learns the components from training data and transforms it.
- `transform()` applies the same learned components to test data.

### D. Visualize the first two components

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Breast Cancer Dataset")
plt.colorbar(label="Original class label")
plt.show()
```

The original labels are used only for coloring the visualization. PCA itself does not use them.

### E. Retain 95% of variance

```python
pca_95 = PCA(n_components=0.95)

X_train_95 = pca_95.fit_transform(X_train_scaled)
X_test_95 = pca_95.transform(X_test_scaled)

print("Components retained:", pca_95.n_components_)
print("Training shape:", X_train_95.shape)
print("Variance retained:",
      pca_95.explained_variance_ratio_.sum())
```

A fractional `n_components` value such as `0.95` tells PCA to retain enough components to explain at least 95% of the training data's variance.

## 7. Important Scikit-Learn Attributes

| Attribute | Purpose |
|---|---|
| `pca.components_` | Directions/weights of the learned components |
| `pca.explained_variance_` | Variance captured by each retained component |
| `pca.explained_variance_ratio_` | Fraction of variance explained by each component |
| `pca.n_components_` | Number of components actually retained |
| `pca.mean_` | Feature means learned by PCA |

## 8. Choosing the Number of Components

Common approaches:

- Keep 2 or 3 components for visualization.
- Choose a variance-retention threshold, such as 90% or 95%.
- Compare downstream model performance for different component counts.

A higher retained-variance percentage does not guarantee better prediction. Evaluate the downstream model using validation data or cross-validation.

## 9. Avoiding Data Leakage

When using PCA in machine learning:

1. Split the dataset first.
2. Fit the scaler on training data only.
3. Fit PCA on scaled training data only.
4. Transform validation and test data using the already fitted scaler and PCA.
5. If tuning PCA and a predictive model together, use a pipeline and cross-validation.

PCA should be fitted inside the training process, not on the entire dataset before splitting.

## 10. Advantages and Limitations

### Advantages
- Reduces dimensionality.
- Can improve visualization.
- Reduces redundancy among correlated features.
- May reduce training time for downstream models.

### Limitations
- Components can be difficult to interpret.
- Discarding components loses some information.
- PCA mainly captures linear structure.
- Outliers can strongly affect the results.
- High variance does not necessarily mean high predictive importance.
- Feature scaling can substantially affect results.

## 11. Key Takeaways

- PCA is an unsupervised dimensionality-reduction technique.
- It creates new features called principal components.
- PC1 captures the greatest variance; later components capture remaining variance.
- Explained variance helps determine how many components to retain.
- Standardization is often useful when features have different scales.
- Fit preprocessing and PCA on training data only.
- Evaluate downstream performance rather than relying only on explained variance.