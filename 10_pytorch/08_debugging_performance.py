
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Create sample regression data
X = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
y = 2 * X + 1

# Create model
model = nn.Linear(1, 1).to(device)

# Inspect tensor shapes and devices
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X device:", X.device)
print("Model device:", next(model.parameters()).device)

# Check whether inputs contain invalid values
print("X is finite:", torch.isfinite(X).all().item())
print("y is finite:", torch.isfinite(y).all().item())

# Forward pass
predictions = model(X)
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, y)

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)   
total_loss = torch.zeros((), device=device)

for epoch in range(100):
    optimizer.zero_grad()

    predictions = model(X)
    loss = loss_fn(predictions, y)

    loss.backward()
    optimizer.step()

    total_loss += loss.detach()

    if (epoch + 1) % 10 == 0:
        print("Average loss:", (total_loss / 10).item())
        total_loss.zero_()
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(
            name,
            "gradient finite:",
            torch.isfinite(parameter.grad).all().item(),
            "gradient norm:",
            parameter.grad.norm().item()
        )
bad_values = torch.tensor([1.0, float("nan"), float("inf")], device=device)

print("Values:", bad_values)
print("Finite mask:", torch.isfinite(bad_values))
print("All finite:", torch.isfinite(bad_values).all().item())

print("Prediction shape:", predictions.shape)
print("Loss:", loss.item())
print("Loss is finite:", torch.isfinite(loss).item())
