
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Generate sample data
X, _ = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# 2. Scale the features
X_scaled = StandardScaler().fit_transform(X)

# 3. Build the hierarchy for the dendrogram
Z = linkage(X_scaled, method="ward")

# plt.figure(figsize=(10, 5))
# dendrogram(Z)
# plt.title("Hierarchical Clustering Dendrogram")
# plt.xlabel("Data points")
# plt.ylabel("Merge distance")
# plt.show()

# 4. Form three clusters
model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)

# 5. Visualize the clusters
plt.scatter(
    X_scaled[:, 1],
    X_scaled[:, 0],
    c=labels,
    s=70
)

plt.title("Agglomerative Clustering")
plt.xlabel("Scaled feature 1")
plt.ylabel("Scaled feature 2")
plt.show()

print("First 10 cluster labels:", labels[:10])
