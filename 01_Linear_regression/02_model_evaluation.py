from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


X = [[1], [2], [3]]
y = [20, 30, 40]


model = LinearRegression()
model.fit(X, y)


predictions = model.predict(X)


mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)

print("Actual values:", y)
print("Predictions:", predictions)
print("MAE:", mae)
print("MSE:", mse)