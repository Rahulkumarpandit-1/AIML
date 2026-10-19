# t-SNE (t-distributed Stochastic Neighbor Embedding)

## 1. What is t-SNE?

t-SNE is a nonlinear dimensionality-reduction technique mainly used to visualize high-dimensional data in 2D or 3D.

For example, a dataset with 30 features can be represented using two coordinates for visualization.

**Main purpose:** Explore patterns, neighborhoods, and possible clusters in high-dimensional data.

## 2. How t-SNE works

1. It measures similarities between data points in the original feature space.
2. Nearby points receive higher similarity; distant points generally receive lower similarity.
3. It constructs a low-dimensional representation, usually in 2D.
4. It adjusts the positions to preserve local neighborhood relationships as well as possible.

t-SNE is an optimization algorithm; we normally use a library implementation rather than calculate its probabilities manually.

## 3. Important parameters

- `n_components=2`: produces a two-dimensional embedding.
- `perplexity=30`: controls the effective neighborhood scale.
- `learning_rate="auto"`: lets scikit-learn select a learning rate.
- `init="pca"`: initializes the embedding using PCA.
- `random_state=42`: makes the randomized process reproducible.

Perplexity does not directly specify the number of points in a cluster. Its appropriate value depends on the dataset and should be tested.

## 4. Example

Suppose a dataset contains 569 samples and 30 features.

Original shape: `(569, 30)`

After applying t-SNE with two components: `(569, 2)`

Each sample now has two coordinates suitable for plotting.

## 5. Scikit-learn implementation

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

data = load_breast_cancer()
X = data.data
y = data.target

# Scale features before computing similarities
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a two-dimensional embedding
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Embedding shape:", X_tsne.shape)

# Visualize using the known class labels
plt.figure(figsize=(9, 6))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="coolwarm",
    alpha=0.7
)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization")
plt.colorbar(label="Class (0 = malignant, 1 = benign)")
plt.show()
```

## 6. Understanding the output

- `fit_transform(X_scaled)` learns the embedding and returns two coordinates per sample.
- `X_tsne[:, 0]` selects the first coordinate.
- `X_tsne[:, 1]` selects the second coordinate.
- `c=y` colors points according to their known classes. The labels are not used to learn the embedding in this example.

Standard scikit-learn t-SNE does not provide the ordinary `transform()` method for embedding new, unseen samples into an already fitted map.

## 7. t-SNE vs PCA

| Property | PCA | t-SNE |
|---|---|---|
| Type | Linear | Nonlinear |
| Main purpose | Dimensionality reduction and compression | Visualization |
| Preserves | Directions of maximum variance | Local neighborhood relationships |
| New samples | Supports `transform()` | Standard implementation lacks ordinary `transform()` |
| Typical use | Feature reduction before modeling | Exploring high-dimensional data |

## 8. Advantages

- Helps visualize high-dimensional data.
- Can reveal local patterns and neighborhoods.
- Can represent nonlinear structures that PCA may not reveal clearly.

## 9. Limitations

- Can be computationally expensive on large datasets.
- Results can change with perplexity, initialization, and other settings.
- Distances between separate groups and apparent cluster sizes may be misleading.
- A visually separated plot does not prove that real classes or clusters exist.
- It is generally not the first choice for transforming features for a predictive model.

## 10. Important practical rules

1. Scale numerical features when their scales differ substantially.
2. Try multiple perplexity values and compare the results.
3. Use t-SNE mainly for exploratory visualization.
4. Do not interpret plot distances as exact original-space distances.
5. Do not use target labels to fit t-SNE when you want an unsupervised embedding.
