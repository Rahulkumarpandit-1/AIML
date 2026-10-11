
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Features: [study hours, attendance percentage]
X = np.array([
    [1, 40], [2, 45], [2.5, 50], [3, 55],
    [3.5, 60], [4, 65], [4.5, 68], [5, 70],
    [5.5, 75], [6, 78], [6.5, 80], [7, 85],
    [7.5, 88], [8, 90], [9, 95], [10, 98]
])

# Target: 0 = Fail, 1 = Pass
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 1, 1, 1
])

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model=make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("X_test:")
print(X_test)

print("\nActual labels:")
print(y_test)

print("\nPredicted labels:")
print(predictions)

print("\nAccuracy:", accuracy_score(y_test, predictions))