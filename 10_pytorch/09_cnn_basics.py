import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10)
).to(device)

# Batch of 8 RGB images, each 32 x 32 pixels
X = torch.randn(8, 3, 32, 32, device=device)

with torch.no_grad():
    predictions = model(X)

print("Input shape:", X.shape)
print("Output shape:", predictions.shape)
print("Predictions for first image:", predictions[0])