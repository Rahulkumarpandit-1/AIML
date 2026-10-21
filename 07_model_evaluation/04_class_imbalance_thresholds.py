from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

# Split off the test set first
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a validation set from the training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# Select threshold using validation probabilities
val_prob = model.predict_proba(X_val)[:, 1]

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    val_pred = (val_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(
        y_val, val_pred, zero_division=0
    ))
    print("Recall:", recall_score(
        y_val, val_pred, zero_division=0
    ))
    print("F1:", f1_score(
        y_val, val_pred, zero_division=0
    ))

# Example final evaluation using the default threshold
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

print("\nFinal test accuracy:", accuracy_score(y_test, test_pred))
print("Final test confusion matrix:")
print(confusion_matrix(y_test, test_pred))