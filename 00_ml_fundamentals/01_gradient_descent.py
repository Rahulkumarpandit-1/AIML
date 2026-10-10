
# 01_gradient_descent.py

# We want to find the value of theta that minimizes:
# J(theta) = (theta - 3)^2

theta = 0.0
learning_rate = 1.0
iterations = 20

for i in range(iterations):
    loss = (theta - 3) ** 2
    gradient = 2 * (theta - 3)

    theta = theta - learning_rate * gradient

    print(
        f"Iteration {i + 1}: "
        f"theta = {theta:.4f}, "
        f"loss before update = {loss:.4f}"
    )

print("Final theta:", round(theta, 4))
