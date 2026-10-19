from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate nonlinear classification data
X, y = make_moons(
    n_samples=500,
    noise=0.25,
    random_state=42
)

# Keep test data separate from training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Linear SVM": SVC(kernel="linear", C=1.0),
    "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale")
}

for name, svm in models.items():
    # Scaling is important for distance-based kernel calculations
    model = make_pipeline(
        StandardScaler(),
        svm
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"{name}: {accuracy:.3f}")
