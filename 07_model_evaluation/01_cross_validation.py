
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Keep the test set untouched until final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Put preprocessing and model inside one pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("Fold scores:", scores)
print("Mean CV accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Final evaluation on the held-out test set
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
