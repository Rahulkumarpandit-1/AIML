import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Before PCA:", X_train_scaled.shape)
print("After PCA:", X_train_pca.shape)

print("Explained variance ratio:",
      pca.explained_variance_ratio_)

print("Total variance retained:",
      pca.explained_variance_ratio_.sum())
pca_95 = PCA(n_components=0.85)

X_train_95 = pca_95.fit_transform(X_train_scaled)
X_test_95 = pca_95.transform(X_test_scaled)

print("Original features:", X_train_scaled.shape[1])
print("Components retained:", pca_95.n_components_)
print("Training shape:", X_train_95.shape)
print("Variance retained:",
      pca_95.explained_variance_ratio_.sum())
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