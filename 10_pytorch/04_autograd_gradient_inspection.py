import torch
import torch.nn as nn

torch.manual_seed(42)

# 1. Create a simple model
model = nn.Linear(1, 1)

# 2. Prepare data: target is approximately y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])

# 3. Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Forward pass
predictions = model(X)
loss = loss_fn(predictions, y_true)

# 5. Calculate gradients
optimizer.zero_grad()
loss.backward()

print("Loss before update:", loss.item())

for name, parameter in model.named_parameters():
    print(f"{name} gradient:", parameter.grad)

# 6. Update model parameters
optimizer.step()

# 7. Check the updated parameters
print("\nParameters after update:")
for name, parameter in model.named_parameters():
    print(name, parameter.data)