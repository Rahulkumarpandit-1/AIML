import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error 
import math
import matplotlib.pyplot as plt
model=LinearRegression()

data = pd.DataFrame({
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [25000, 30000, 35000, 42000, 48000, 55000, 61000, 68000, 75000, 83000]
})

X=data[["experience"]]
y=data["salary"]    

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
new_data=pd.DataFrame({
    "experience":[7]
})

model.fit(X_train,y_train)

prediction=model.predict(new_data)

plt.scatter(X,y )

print("Prediction:")
print(prediction)
plt.show()
