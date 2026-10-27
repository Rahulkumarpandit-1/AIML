
import torch
import torch.nn as nn
import copy

# Training examples
X_train = torch.tensor([
    [1.0], [2.0], [3.0], [4.0], [5.0]
])

y_train = torch.tensor([
    [3.0], [5.0], [7.0], [9.0], [11.0]
])

# Validation examples
X_val = torch.tensor([
    [6.0], [7.0], [8.0]
])

y_val = torch.tensor([
    [13.0], [15.0], [17.0]
])

# Model
model = nn.Linear(1, 1)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# Early stopping settings
best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())

patience = 5
epochs_without_improvement = 0
max_epochs = 100

for epoch in range(max_epochs):

    # 1. Training
    model.train()

    prediction = model(X_train)
    loss = loss_fn(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 2. Validation
    model.eval()

    with torch.no_grad():
        val_prediction = model(X_val)
        val_loss = loss_fn(val_prediction, y_val)

    current_val_loss = val_loss.item()

    # 3. Check whether validation loss improved
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # 4. Display progress
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Training loss = {loss.item():.4f}, "
            f"Validation loss = {current_val_loss:.4f}"
        )

    # 5. Stop if patience is exhausted
    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

# 6. Restore the best model parameters
model.load_state_dict(best_state)

print("\nBest validation loss:", best_val_loss)
print("Best model weight:", model.weight.item())
print("Best model bias:", model.bias.item())
