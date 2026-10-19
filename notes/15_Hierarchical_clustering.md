# Hierarchical Clustering

## 1. What is Hierarchical Clustering?

Hierarchical clustering is an unsupervised learning algorithm that groups similar data points into a hierarchy of clusters.

Unlike K-Means, which directly produces a flat clustering for a chosen K, hierarchical clustering builds a tree-like structure that can be cut at different levels to obtain different numbers of clusters.

**Applications**
- Customer segmentation
- Grouping similar documents
- Biological classification
- Exploring patterns in unlabeled datasets

## 2. Types of Hierarchical Clustering

### A. Agglomerative — Bottom-Up

1. Start with every data point as a separate cluster.
2. Find the two closest clusters.
3. Merge them.
4. Recalculate distances between clusters.
5. Repeat until the stopping condition is reached.

### B. Divisive — Top-Down

1. Start with all points in one cluster.
2. Split a cluster into smaller groups.
3. Continue splitting until the stopping condition is reached.

Agglomerative clustering is our primary focus.

## 3. Agglomerative Clustering Dry Run

Suppose our one-dimensional data points are:

`1, 2, 8, 9`

Initially:

```text
{1}    {2}    {8}    {9}
```

**Step 1: Merge the closest points**

Distance(1, 2) = 1  
Distance(8, 9) = 1

Both pairs can merge:

```text
{1, 2}       {8, 9}
```

**Step 2: Calculate the distance between clusters**

Using single linkage:

Distance between clusters = minimum cross-cluster distance.

Cross-cluster distances:

- Distance(1, 8) = 7
- Distance(1, 9) = 8
- Distance(2, 8) = 6
- Distance(2, 9) = 7

Therefore:

`D({1, 2}, {8, 9}) = min(7, 8, 6, 7) = 6`

The two clusters merge at distance 6.

```text
        {1, 2, 8, 9}
           /     \
       {1, 2}   {8, 9}
        /  \     /  \
       1    2   8    9
```

If we cut the hierarchy below distance 6 but above distance 1, we obtain two clusters: `{1, 2}` and `{8, 9}`.

## 4. Linkage Methods

A linkage method determines how the distance between two clusters is calculated.

### A. Single Linkage

Uses the minimum distance between any point in one cluster and any point in the other.

Formula:

`D(A, B) = min distance(a, b)`

Here, a belongs to cluster A and b belongs to cluster B.

- Can create long, chain-like clusters.
- One chain of nearby points can connect otherwise distant groups.

### B. Complete Linkage

Uses the maximum distance between any cross-cluster pair.

Formula:

`D(A, B) = max distance(a, b)`

- Tends to favor compact clusters.
- Can be sensitive to outliers.

### C. Average Linkage

Uses the average of all cross-cluster distances.

Formula:

`D(A, B) = sum of all cross-cluster distances / (number of points in A × number of points in B)`

- Provides a compromise between single and complete linkage.

### D. Ward Linkage

Chooses the merge that causes the smallest increase in total within-cluster squared error.

- Often produces compact clusters.
- In scikit-learn, Ward linkage requires Euclidean distance.

**Quick revision**

- Single = minimum
- Complete = maximum
- Average = average pairwise distance
- Ward = minimum increase in within-cluster squared error

## 5. What is a Dendrogram?

A dendrogram is a tree diagram that displays the order and distance at which clusters merge.

How to read it:

- The bottom labels represent individual data points.
- Branches show how points and clusters merge.
- The vertical axis represents merge distance or linkage distance.
- A horizontal cut through the tree determines the resulting clusters.

A large gap between merge heights can suggest a useful cut, but it does not guarantee the best clustering.

## 6. Python Practical

File: `machine_learning/06_unsupervised_learning/02_hierarchical_clustering.py`

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# Generate sample data
X, _ = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build hierarchy for dendrogram
Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data points")
plt.ylabel("Merge distance")
plt.show()

# Form three clusters
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)

# Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    s=30
)

plt.title("Agglomerative Clustering")
plt.xlabel("Scaled feature 1")
plt.ylabel("Scaled feature 2")
plt.show()

print("First 10 cluster labels:", labels[:10])
```

### Important code explanations

- `StandardScaler()` standardizes the features.
- `linkage(X_scaled, method="ward")` builds the hierarchy for the dendrogram.
- `dendrogram(Z)` visualizes the merge structure.
- `AgglomerativeClustering(n_clusters=3)` creates a clustering model with three final clusters.
- `fit_predict(X_scaled)` fits the model and returns a cluster ID for every point.

SciPy builds the dendrogram hierarchy, while scikit-learn independently fits the clustering model to produce labels. Here, both use Ward linkage on the same scaled data.

## 7. Limitations

1. Can require substantial computation and memory on large datasets.
2. Results depend on the distance metric and linkage method.
3. Agglomerative merges are generally irreversible.
4. Outliers and feature scales can affect cluster formation.
5. Choosing the number of clusters still requires judgment when cutting the dendrogram.

## 8. K-Means vs Hierarchical Clustering

| K-Means | Hierarchical Clustering |
|---|---|
| Requires K before fitting | Builds a hierarchy that can be cut later |
| Updates centroids | Merges or splits clusters |
| Produces a flat clustering | Produces nested groups |
| Often suitable for larger datasets | Can be expensive for large datasets |

## 9. Quick Revision

- Hierarchical clustering is an unsupervised learning technique.
- Agglomerative clustering works bottom-up.
- Divisive clustering works top-down.
- Linkage determines the distance between clusters.
- A dendrogram shows the merge hierarchy.
- A horizontal cut determines the number of clusters.
- Feature scaling can matter when features have different units.
- Cluster IDs are arbitrary identifiers, not rankings.

**Status:** Learning. Mark this topic Practiced after running the Python code, interpreting the dendrogram, and trying at least two linkage methods.