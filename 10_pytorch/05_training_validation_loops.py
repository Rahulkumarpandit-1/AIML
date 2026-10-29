import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# Dataset: y = 2x + 1
X = torch.arange(1, 11, dtype=torch.float32).reshape(-1, 1)
y = 2 * X + 1

# Training and validation split
X_train, y_train = X[:8], y[:8]
X_val, y_val = X[8:], y[8:]

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=4,
    shuffle=True
)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 50
best_val_loss = float("inf")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * len(X_batch)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_fn(val_predictions, y_val)

    avg_val_loss = val_loss.item()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"train={avg_train_loss:.4f}, "
            f"validation={avg_val_loss:.4f}"
        )

# Restore the best validation checkpoint
model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)

print("\nBest validation loss:", best_val_loss)
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())