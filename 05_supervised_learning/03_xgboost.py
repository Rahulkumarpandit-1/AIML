from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split        
from sklearn.metrics import accuracy_score      
from xgboost import XGBClassifier   


data = load_breast_cancer()
X = data.data
y = data.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model=XGBClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    learning_rate=0.1,
    eval_metric="logloss",
    
    )

model.fit(X_train,y_train)

predictions=model.predict(X_test)

accuracy=accuracy_score(y_test,predictions)
print("XGBoost accuracy:", accuracy)    

