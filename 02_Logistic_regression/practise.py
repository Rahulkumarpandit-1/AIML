import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix ,recall_score,precision_score ,f1_score   
model=LogisticRegression()
data = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "passed": [0, 0, 0, 1, 1, 1, 1, 1]
})
X=data[["hours"]]
y=data["passed"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train,y_train)
print(X_test)
print(y_test)
prediction=model.predict(X_test)
print(prediction)   

probabilities=model.predict_proba(X_test)
print(probabilities)

accuracy=accuracy_score(y_test,prediction)
print("accuracy:",accuracy)     
cm=confusion_matrix(y_test,prediction)
print("cm:",cm)
recall=recall_score(y_test,prediction)
print("recall",recall)
precision=precision_score(y_test,prediction)
print("precision",precision)
f1=f1_score(y_test,prediction)
print("f1:",f1)   