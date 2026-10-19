import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Generate curved clusters
X, _ = make_moons(
    n_samples=500,
    noise=0.06,
    random_state=42
)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
model = DBSCAN(eps=0.25, min_samples=10)
labels = model.fit_predict(X_scaled)

# Count clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
print("First 10 labels:", labels[:10])

# Visualize the results
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