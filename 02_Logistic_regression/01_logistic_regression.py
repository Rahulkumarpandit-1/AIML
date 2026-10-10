
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Feature: hours studied
X = np.array([
    [1], [2], [2.5], [3], [3.5],
    [4], [4.5], [5], [5.5], [6],
    [6.5], [7], [7.5], [8], [9],
    [10]
])

# Target: 0 = Fail, 1 = Pass
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 1, 0, 1,
    1, 1, 1, 1, 1, 1
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Scale the feature and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X_train, y_train)

# Predict class labels and probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("X_test:")
print(X_test)

print("\nActual labels:")
print(y_test)

print("\nPredicted labels:")
print(predictions)

print("\nProbabilities [Fail, Pass]:")
print(probabilities)

print("\nAccuracy:", accuracy_score(y_test, predictions))
