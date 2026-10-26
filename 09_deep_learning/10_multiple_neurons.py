
import torch
import torch.nn as nn

# One training example: two inputs, one target
x = torch.tensor([[2.0, 3.0]])
target = torch.tensor([[1.0]])

# Network: 2 inputs -> 2 hidden neurons -> 1 output
model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)

# Use fixed parameters so the example is reproducible
with torch.no_grad():
    model[0].weight.copy_(torch.tensor([
        [0.5, 0.2],
        [-0.3, 0.4]
    ]))
    model[0].bias.copy_(torch.tensor([-0.5, 0.2]))
    model[2].weight.copy_(torch.tensor([[2.0, -1.0]]))
    model[2].bias.copy_(torch.tensor([0.5]))

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# One training step
prediction = model(x)
loss = loss_fn(prediction, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Prediction before update:", prediction.item())
print("Loss before update:", loss.item())
print("Prediction after update:", model(x).item())
