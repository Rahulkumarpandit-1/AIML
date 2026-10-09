from sklearn.tree import DecisionTreeClassifier         
import pandas as pd
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt   
model=DecisionTreeClassifier()

df=pd.DataFrame({
    "city":[1,2,3,4,5,6,7,8]
})
data = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "passed": [0, 0, 0, 1, 1, 1, 1, 1]
})

X = data[["hours"]]
y = data["passed"]
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train,y_train)
predictions = model.predict(X_test)
