
# 03_adam.py

theta = 0.0
learning_rate = 0.1

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

m = 0.0
v = 0.0

for t in range(1, 21):
    # J(theta) = (theta - 3)^2
    gradient = 2 * (theta - 3)
    loss = (theta - 3) ** 2

    # Moving averages of gradient and squared gradient
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2

    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # Parameter update
    theta = theta - learning_rate * m_hat / (v_hat**0.5 + epsilon)

    print(f"Iteration {t}: theta={theta:.4f}, loss={loss:.4f}")

print("Final theta:", round(theta, 4))
