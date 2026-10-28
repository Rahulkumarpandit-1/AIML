import torch
import torch.nn as nn

layer = nn.Linear(1, 8)

print("Weight shape:", layer.weight.shape)
print("Bias shape:", layer.bias.shape)
print("Total parameters:", sum(p.numel() for p in layer.parameters()))

X = torch.tensor([[2.0], [4.0], [6.0]])
output = layer(X)

print("Input shape:", X.shape)
print("Output shape:", output.shape)