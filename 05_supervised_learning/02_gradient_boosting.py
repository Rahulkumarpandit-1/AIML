from sklearn.ensemble import GradientBoostingRegressor

# Input: house size in square feet
X = [[500], [700], [900], [1100], [1300], [1500]]

# Target: house price in lakh rupees
y = [20, 25, 32, 38, 45, 52]

# Create the model
model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=1,
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict the price of a 1000-square-foot house
prediction = model.predict([[1000]])

print("Predicted price:", prediction[0], "lakh")