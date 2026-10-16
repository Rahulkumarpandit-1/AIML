from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tree=DecisionTreeClassifier(random_state=42)
tree.fit(X_train,y_train)
tree_predictions=tree.predict(X_test)


forest=RandomForestClassifier(n_estimators=100,random_state=42)
forest.fit(X_train,y_train)
forest.predictions=forest.predict(X_test)   
importances = forest.feature_importances_

for i in importances.argsort()[::-1][:5]:
    print(data.feature_names[i], importances[i])

print("Tree accuracy:",accuracy_score(y_test,tree_predictions)) 

print("Forest accuracy:",accuracy_score(y_test,forest.predictions)) 
