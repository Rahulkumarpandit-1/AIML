
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.75,
    random_state=42
)

# Test different numbers of clusters
inertias = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)
    inertias.append(model.inertia_)

# Plot the elbow curve
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means on scaled data
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)

print("Silhouette score:", silhouette_score(X_scaled, labels))
print("Scaled centroids:\n", model.cluster_centers_)
