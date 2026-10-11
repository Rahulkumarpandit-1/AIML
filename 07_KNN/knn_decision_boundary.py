
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. Create a two-class dataset
X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=42
)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create a grid covering the feature space
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# 4. Compare different values of K
k_values = [1, 3, 5, 15]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, k in zip(axes.ravel(), k_values):
    model = Pipeline([
        ("standardscaler", StandardScaler()),
        ("kneighborsclassifier", KNeighborsClassifier(n_neighbors=k))
    ])

    # Train only on training data
    model.fit(X_train, y_train)

    # Predict the class of every grid point
    predictions = model.predict(grid).reshape(xx.shape)

    # Draw the decision regions
    ax.contourf(xx, yy, predictions, alpha=0.3, cmap="coolwarm")

    # Show the training samples
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolors="black",
        s=25
    )

    ax.set_title(f"K = {k} | Test accuracy: {model.score(X_test, y_test):.2f}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
