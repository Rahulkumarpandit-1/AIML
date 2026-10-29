import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def evaluate():
    with torch.no_grad():
        predictions = model(X)
        return loss_fn(predictions, y).item()


initial_loss = evaluate()

for epoch in range(500):
    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

final_loss = evaluate()

with torch.no_grad():
    test_prediction = model(torch.tensor([[5.0]])).item()

print("Initial loss:", initial_loss)
print("Final loss:", final_loss)
print("Prediction for x=5:", test_prediction)
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())




# import torch
# import torch.nn as nn

# # Training examples
# X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# # 1. Build the model
# model = nn.Linear(1, 1)

# # 2. Choose a loss function and optimizer
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# # 3. Train
# for epoch in range(500):
#     predictions = model(X)
#     loss = loss_fn(predictions, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # 4. Test the trained model
# with torch.no_grad():
#     test_input = torch.tensor([[5.0]])
#     prediction = model(test_input)

# print("Final loss:", loss.item())
# print("Prediction for x=5:", prediction.item())

# import torch
# import torch.nn as nn


# class StudyModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(1, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1)
#         )

#     def forward(self, x):
#         return self.layers(x)


# model = StudyModel()

# # Inspect all trainable parameters
# for name, parameter in model.named_parameters():
#     print(name, parameter.shape, parameter.numel())

# total_parameters = sum(
#     parameter.numel()
#     for parameter in model.parameters()
# )

# print("Total trainable parameters:", total_parameters)

# # Test the model
# X = torch.tensor([[1.0], [2.0], [3.0]])
# predictions = model(X)

# print("Input shape:", X.shape)
# print("Output shape:", predictions.shape)
#--------------------------------------------------------------------------------
# import torch
# import torch.nn as nn

# model = nn.Linear(1, 1)

# X = torch.tensor([[2.0], [4.0], [6.0]])
# predictions = model(X)

# print("Predictions before training:")
# print(predictions.detach())

# print("Trainable parameters:")
# for name, parameter in model.named_parameters():
#     print(name, parameter.detach())