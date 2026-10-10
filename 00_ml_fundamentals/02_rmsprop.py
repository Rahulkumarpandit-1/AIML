    
# 02_rmsprop.py

theta = 0.0
learning_rate = 0.1
rho = 0.9
epsilon = 1e-8
s = 0.0

for i in range(20):
    # J(theta) = (theta - 3)^2
    gradient = 2 * (theta - 3)
    loss = (theta - 3) ** 2

    # Moving average of squared gradients
    s = rho * s + (1 - rho) * gradient**2

    # RMSProp parameter update
    theta = theta - (learning_rate / (s**0.5 + epsilon)) * gradient

    print(
        f"Iteration {i + 1}: "
        f"theta = {theta:.4f}, "
        f"loss = {loss:.4f}"
    )

print("Final theta:", round(theta, 4))
