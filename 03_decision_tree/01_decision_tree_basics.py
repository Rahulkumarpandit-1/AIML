from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Features: [hours_studied]
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Labels: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Predict for a student who studied 6 hours
print("Prediction for 6 hours:", model.predict([[6]]))