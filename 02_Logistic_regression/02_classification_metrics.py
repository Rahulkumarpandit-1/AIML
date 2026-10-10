

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Actual labels from your previous run
y_test = [0, 1, 0, 1]

# Predictions from your previous run
predictions = [0, 1, 0, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)
