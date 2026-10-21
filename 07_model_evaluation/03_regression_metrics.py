from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

y_actual = [20, 30, 40]
y_predicted = [22, 27, 41]

mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
rmse = root_mean_squared_error(y_actual, y_predicted)
r2 = r2_score(y_actual, y_predicted)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)