
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Create dataset on CPU for DataLoader
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

model = nn.Linear(1, 1).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True
) as prof:
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        prediction = model(X_batch)
        loss = loss_fn(prediction, y_batch)

        loss.backward()
        optimizer.step()

print("Last batch loss:", loss.item())
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=5
    )
)