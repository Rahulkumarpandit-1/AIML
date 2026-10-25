# Minimize loss(w) = (w - 3) ** 2

w = 0.0
learning_rate = 0.1

for step in range(10):
    loss = (w - 3) ** 2
    gradient = 2 * (w - 3)

    w = w - learning_rate * gradient

    new_loss = (w - 3) ** 2
    print(f"Step {step + 1}: weight={w:.3f}, loss={new_loss:.3f}")