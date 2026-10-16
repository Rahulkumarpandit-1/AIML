from sklearn.tree import DecisionTreeClassifier ,export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split        

# Input feature: study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Target: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

model=DecisionTreeClassifier(max_depth=2,random_state=42)

model.fit(X_train,y_train)

new_students=[[2],[5]]
predictions=model.predict(new_students)

print(y_test)
print(predictions)
print("Accuracy_score:",accuracy_score(y_test,predictions))
print(export_text(model,feature_names=["study_hours"]))