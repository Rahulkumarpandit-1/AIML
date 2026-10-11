from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

predictions = model.predict(X)

print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y, predictions))
print("Confusion matrix:")
print(confusion_matrix(y, predictions))