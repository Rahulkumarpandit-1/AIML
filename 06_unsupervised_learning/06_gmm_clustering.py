import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Generate sample data
X, _ = make_blobs(
    n_samples=400,
    centers=3,
    cluster_std=[1.0, 1.5, 0.8],
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train GMM
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42,
    n_init=5
)

gmm.fit(X_scaled)

# Predict hard cluster labels
labels = gmm.predict(X_scaled)

# Predict soft cluster probabilities
probabilities = gmm.predict_proba(X_scaled)

print("Cluster labels:", labels[:10])
print("Probabilities for first point:", probabilities[0])
print("Probability sum:", probabilities[0].sum())
print("Learned mixture weights:", gmm.weights_)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Gaussian Mixture Model Clustering")
plt.show()