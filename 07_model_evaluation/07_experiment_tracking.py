
import csv
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Reproducible split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Define model settings for this experiment
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=2,
    random_state=42
)

# Cross-validation on training data
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

# Fit on all training data, then evaluate on test data
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

# Record experiment details
experiment = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model": "DecisionTreeClassifier",
    "max_depth": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "cv_folds": 5,
    "cv_mean_accuracy": round(cv_scores.mean(), 4),
    "cv_std_accuracy": round(cv_scores.std(), 4),
    "test_accuracy": round(test_accuracy, 4)
}

# Append results to CSV
filename = "experiment_log.csv"

with open(filename, "a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=experiment.keys()
    )

    if file.tell() == 0:
        writer.writeheader()

    writer.writerow(experiment)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("CV standard deviation:", cv_scores.std())
print("Test accuracy:", test_accuracy)
print("Experiment recorded in:", filename)
