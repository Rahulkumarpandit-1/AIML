from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load a real classification dataset
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

# Reserve the test set for final evaluation
X_remaining, X_test, y_remaining, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.25,
    random_state=42,
    stratify=y_remaining
)

alphas = [0.0, 0.001, 0.005, 0.01, 0.02]
results = []

# Select a pruning value using validation accuracy
for alpha in alphas:
    model = DecisionTreeClassifier(
        ccp_alpha=alpha,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, model.predict(X_train)
    )
    val_accuracy = accuracy_score(
        y_val, model.predict(X_val)
    )

    results.append((alpha, val_accuracy, model))

    print(
        f"alpha={alpha:.3f} | "
        f"depth={model.get_depth()} | "
        f"leaves={model.get_n_leaves()} | "
        f"train={train_accuracy:.3f} | "
        f"validation={val_accuracy:.3f}"
    )

best_alpha, best_val_accuracy, best_model = max(
    results, key=lambda result: result[1]
)

print("\nSelected alpha:", best_alpha)
print("Validation accuracy:", best_val_accuracy)

# Evaluate the selected model on the test set
test_predictions = best_model.predict(X_test)
print(
    "Final test accuracy:",
    accuracy_score(y_test, test_predictions)
)

# Inspect the most important features
importances = best_model.feature_importances_
ranked_indices = importances.argsort()[::-1]

print("\nTop 10 features:")
for index in ranked_indices[:10]:
    print(f"{feature_names[index]}: {importances[index]:.4f}")