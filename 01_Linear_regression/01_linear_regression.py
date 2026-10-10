from sklearn.linear_model import LinearRegression


X = [[1], [2], [3]]
y = [20, 30, 40]

model=LinearRegression()

model.fit(X,y)

print("Weight:", model.coef_[0])
print("Intercept:", model.intercept_)


prediction = model.predict([[4]])
print("Predicted marks:", prediction[0])