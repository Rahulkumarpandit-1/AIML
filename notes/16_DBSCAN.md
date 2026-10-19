# DBSCAN Clustering

## 1. What Is DBSCAN?

**DBSCAN** stands for *Density-Based Spatial Clustering of Applications with Noise*.

It is an unsupervised clustering algorithm that groups closely packed data points and identifies isolated points as noise or outliers.

Unlike K-Means, DBSCAN does not require us to specify the number of clusters beforehand.

![DBSCAN clustering visualization showing clusters and noise points](PASTE_DIRECT_IMAGE_URL_HERE)

### Why use DBSCAN?

- Identifies clusters with irregular or curved shapes.
- Detects noise and outliers.
- Does not require the number of clusters as an input.
- Works well when clusters form sufficiently dense regions.

---

## 2. Important Parameters

DBSCAN uses two main parameters.

### `eps` (Epsilon)

The maximum distance between two points for them to be considered neighbors.

- Small `eps`: fewer neighbors; clusters may fragment and more points may become noise.
- Large `eps`: more neighbors; separate clusters may merge.

### `min_samples`

The minimum number of points required within a point's `eps` neighborhood for it to qualify as a core point.

In scikit-learn, the point itself counts toward `min_samples`.

Example:

```python
model = DBSCAN(eps=0.25, min_samples=5)
```

This means the neighborhood radius is 0.25 in the feature space, and a point needs at least five points in its neighborhood, including itself, to be a core point.

**Important:** The appropriate `eps` value depends on the distance scale of the data, which is why feature scaling often matters.

---

## 3. Three Types of Points

### A. Core Point

A point with at least `min_samples` points in its `eps` neighborhood, including itself.

Core points help form and expand clusters.

### B. Border Point

A point that does not meet the core-point density requirement but lies within the neighborhood of a core point.

It can be included in a cluster.

### C. Noise Point

A point that is neither a core point nor reachable from a core point. It is not assigned to any cluster.

In scikit-learn, noise points receive the label `-1`.

![DBSCAN core points, border points, epsilon neighborhood and noise points](https://images.openai.com/static-rsc-4/em6Y7xxwvNPKwa7hezU5x9e4RDUySCbwyX1Qvoxv9eBUjGL7lBVvqSO1IZVNx6nHulL5ZLz1hrZhZIWkEQmZDtQ23ji8z1Me_0m56bspzEfEyB00tQS7UqhAef8v5mYzdGmh51GOILH6RJIIZUarm1H1Sfzkvu2Glaa3eydhZEbSoxmLeEK7epbOyFVI1k3v?purpose=fullsize)

---

## 4. How DBSCAN Works

1. Select an unprocessed point.
2. Find all points within its `eps` neighborhood.
3. Check whether the neighborhood contains at least `min_samples` points.
4. If it qualifies as a core point, start or expand a cluster.
5. Add reachable neighboring points to that cluster and continue exploring core points.
6. If a point cannot be assigned to any cluster, label it as noise.

![DBSCAN algorithm workflow showing core point detection and cluster expansion](PASTE_DIRECT_IMAGE_URL_HERE)

---

## 5. Understanding the Parameters Experimentally

### Effect of `eps`

| `eps` | Typical effect |
|---|---|
| Small | More fragmented clusters or noise |
| Moderate | May separate dense regions effectively |
| Large | Clusters may merge; noise may decrease |

### Effect of `min_samples`

| `min_samples` | Typical effect |
|---|---|
| Small | Less density required to form clusters |
| Large | Stricter density requirement; more points may become noise |

These are general trends, not guarantees. The actual outcome depends on the dataset.

---

## 6. Python Implementation

File: `machine_learning/06_unsupervised_learning/03_dbscan_clustering.py`

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Generate a curved dataset
X, _ = make_moons(
    n_samples=500,
    noise=0.06,
    random_state=42
)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
model = DBSCAN(eps=0.25, min_samples=5)
labels = model.fit_predict(X_scaled)

# Calculate the number of clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
print("First 10 labels:", labels[:10])

# Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=25
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

### Understanding the output

- `fit_predict(X_scaled)` fits DBSCAN and returns a cluster label for every data point.
- `0`, `1`, and other non-negative integers represent cluster IDs.
- `-1` represents noise.
- The cluster IDs are arbitrary; cluster `1` is not better than cluster `0`.

---

## 7. Why Feature Scaling Matters

DBSCAN relies on distances between points.

Suppose one feature represents income, ranging from 10,000 to 100,000, while another represents a score ranging from 0 to 10.

Without scaling, income can dominate the distance calculation.

Standardization transforms each feature approximately to:

**Standardized value = (value − feature mean) / feature standard deviation**

This helps features with different units contribute more comparably to Euclidean distance.

Scaling is not mandatory for every dataset, but it is often useful when feature scales differ.

---

## 8. Limitations of DBSCAN

- **Parameter sensitivity:** Choosing a suitable `eps` can be difficult.
- **Different densities:** One `eps` value may not work well for clusters with very different densities.
- **High dimensions:** Distances can become less informative as dimensionality increases.
- **Distance and scaling:** Poor distance choices or unscaled features can produce misleading clusters.
- **Small or sparse clusters:** These may be classified as noise if they do not satisfy the density requirements.

---

## 9. Comparison with Other Clustering Algorithms

| Property | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Number of clusters specified beforehand | Yes | Often, when cutting the hierarchy | No |
| Detects noise explicitly | No | Not by default | Yes |
| Irregular cluster shapes | Often struggles | Depends on linkage | Can handle them well |
| Main controls | `n_clusters` | Linkage and cut/cluster count | `eps`, `min_samples` |
| Different cluster densities | Can struggle | Depends on linkage | Can struggle |

---

## 10. Practical Experiments

Keep the dataset and `random_state` unchanged to make comparisons fair.

### Experiment A: Change `eps`

Try these settings one at a time:

```python
DBSCAN(eps=0.15, min_samples=5)
DBSCAN(eps=0.25, min_samples=5)
DBSCAN(eps=0.40, min_samples=5)
```

Record the number of clusters, noise points, and changes in the plot.

### Experiment B: Change `min_samples`

Keep `eps=0.25`:

```python
DBSCAN(eps=0.25, min_samples=3)
DBSCAN(eps=0.25, min_samples=5)
DBSCAN(eps=0.25, min_samples=10)
```

Observe how changing the density requirement affects the results.

### Experiment log

| `eps` | `min_samples` | Clusters | Noise points |
|---:|---:|---:|---:|
| 0.15 | 5 | Record your output | Record your output |
| 0.25 | 5 | Record your output | Record your output |
| 0.40 | 5 | Record your output | Record your output |

Replace the placeholders with your actual results rather than guessing.

---

## 11. Key Takeaways

- DBSCAN is a density-based clustering algorithm.
- `eps` defines the neighborhood distance.
- `min_samples` controls the density required for a core point.
- Core points expand clusters; border points can join clusters; noise points remain unassigned.
- In scikit-learn, `-1` indicates noise.
- Scaling and parameter selection can significantly affect the results.
- DBSCAN can identify irregular cluster shapes but may struggle with varying densities.