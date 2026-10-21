from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("Brier score:", brier_score_loss(y_test, prob))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

CalibrationDisplay.from_predictions(
    y_test, prob, n_bins=10, strategy="uniform"
)
plt.title("Calibration Curve")
plt.show()

# Inspect individual errors
for i in range(min(10, len(y_test))):
    if pred[i] != y_test[i]:
        print(
            f"Index: {i}, Actual: {y_test[i]}, "
            f"Predicted: {pred[i]}, Probability: {prob[i]:.3f}"
        )