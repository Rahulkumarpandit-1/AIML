
import torch

x = torch.tensor(2.0)
target = torch.tensor(6.0)

w = torch.tensor(1.0, requires_grad=True)
learning_rate = 0.1

for step in range(10):
    # 1. Forward propagation
    prediction = w * x

    # 2. Calculate loss
    loss = (prediction - target) ** 2

    # 3. Calculate gradient
    loss.backward()

    # 4. Update weight
    with torch.no_grad():
        w -= learning_rate * w.grad

    # 5. Clear the old gradient
    w.grad.zero_()

    print(
        f"Step {step + 1}: "
        f"Weight={w.item():.4f}, "
        f"Loss={loss.item():.4f}"
    )
