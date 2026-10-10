from sklearn.linear_model import LinearRegression

# Features: [study hours, attendance percentage]
X = [
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 85],
]

# Target: marks
y = [35, 40, 50, 55, 65, 70]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Display learned parameters
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

# Predict marks for 4 study hours and 78% attendance
prediction = model.predict([[4, 78]])
print("Predicted marks:", prediction[0])