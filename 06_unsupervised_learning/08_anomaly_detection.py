
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Generate ordinary observations
X_normal, _ = make_blobs(
    n_samples=300,
    centers=2,
    cluster_std=0.8,
    random_state=42
)

# Add deliberately unusual observations
rng = np.random.default_rng(42)
X_outliers = rng.uniform(low=-6, high=6, size=(15, 2))

X = np.vstack([X_normal, X_outliers])

# Fit Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

labels = model.fit_predict(X)

# Separate observations by predicted label
inliers = X[labels == 1]
outliers = X[labels == -1]

print("Total observations:", len(X))
print("Predicted inliers:", len(inliers))
print("Predicted anomalies:", len(outliers))

# Visualize the predictions
plt.scatter(
    inliers[:, 0], inliers[:, 1],
    label="Predicted inliers"
)
plt.scatter(
    outliers[:, 0], outliers[:, 1],
    label="Predicted anomalies"
)

plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
