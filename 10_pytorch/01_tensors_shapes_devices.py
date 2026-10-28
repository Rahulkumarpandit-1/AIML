import torch

x = torch.tensor([1, 2, 3, 4, 5, 6])

x = x.float()
y = x.reshape(2, 3)

print("Shape:", y.shape)
print("Dimensions:", y.ndim)
print("Elements:", y.numel())
print("Data type:", y.dtype)
print("First row:", y[0])