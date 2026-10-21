
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Keep the test set untouched during tuning
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Define cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Define hyperparameters to test
param_grid = {
    "max_depth": [2, 3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# 5. Grid Search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

print("Best Grid Search parameters:", grid_search.best_params_)
print("Best mean CV accuracy:", grid_search.best_score_)

# 6. Randomized Search
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=15,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True
)

random_search.fit(X_train, y_train)

print("\nBest Random Search parameters:", random_search.best_params_)
print("Best mean CV accuracy:", random_search.best_score_)

# 7. Evaluate the selected models on the test set
grid_pred = grid_search.best_estimator_.predict(X_test)
random_pred = random_search.best_estimator_.predict(X_test)

print("\nGrid Search test accuracy:",
      accuracy_score(y_test, grid_pred))
print("Random Search test accuracy:",
      accuracy_score(y_test, random_pred))

print("\nGrid Search classification report:")
print(classification_report(y_test, grid_pred))
