# PyTorch: `torch.nn` and Model Building

## 1. What is `torch.nn`?

`torch.nn` provides building blocks for neural networks, including layers, activation functions, loss functions, and the `Module` base class.

```python
import torch
import torch.nn as nn
```

## 2. The `nn.Module` Class

Most custom PyTorch models inherit from `nn.Module`.

```python
class StudyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)
```

- `nn.Module`: base class for models and layers.
- `__init__()`: defines and registers layers and parameters.
- `super().__init__()`: initializes the parent module.
- `forward()`: defines how input flows through the model.
- `model(X)`: invokes the model's forward computation.

Normally, call `model(X)` rather than calling `model.forward(X)` directly.

## 3. Linear Layers

`nn.Linear(in_features, out_features)` applies a learned linear transformation:

`output = input @ weight.T + bias`

For `nn.Linear(1, 8)`:
- Input features: 1
- Output features: 8
- Weight shape: `(8, 1)`
- Bias shape: `(8,)`
- Trainable parameters: `8 × 1 + 8 = 16`

PyTorch stores the weight tensor in the shape `(out_features, in_features)`.

### Parameter count

For a linear layer:

`Number of parameters = in_features × out_features + out_features`

The final term accounts for the bias values.

## 4. Tensor Shapes

A tensor with shape `(batch_size, in_features)` becomes `(batch_size, out_features)` after a linear layer.

Example:

`(3, 1) → Linear(1, 8) → (3, 8)`

`(3, 8) → ReLU → (3, 8)`

`(3, 8) → Linear(8, 1) → (3, 1)`

The batch dimension stays unchanged. ReLU changes values but not tensor shape.

## 5. Activation Functions

`nn.ReLU()` applies the rectified linear unit activation:

`ReLU(x) = max(0, x)`

It introduces nonlinearity and has no trainable parameters.

## 6. Inspecting Parameters

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.numel())
```

- `named_parameters()`: returns parameter names and tensors.
- `shape`: gives the dimensions of a tensor.
- `numel()`: counts all elements in a tensor.
- `parameters()`: iterates over the model's parameters.

For a model containing `Linear(1, 8)`, ReLU, and `Linear(8, 1)`:

- First linear layer: 16 parameters.
- ReLU: 0 parameters.
- Second linear layer: 9 parameters.
- Total: 25 trainable parameters.

## 7. Building vs Training

Building a model defines its architecture and initializes its parameters. Training adjusts the parameters using examples and a learning algorithm.

Typical training sequence:

1. Forward pass: `predictions = model(X)`
2. Calculate loss: `loss = loss_fn(predictions, y)`
3. Clear previous gradients: `optimizer.zero_grad()`
4. Backpropagate: `loss.backward()`
5. Update parameters: `optimizer.step()`

Each step has a distinct responsibility. Calling `model(X)` alone does not train the model.

## 8. A Minimal Training Example

```python
import torch
import torch.nn as nn

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    prediction = model(torch.tensor([[5.0]]))

print("Final loss:", loss.item())
print("Prediction for x=5:", prediction.item())
```

This example learns an approximation of `y = 2x`.

- `nn.MSELoss()` measures mean squared error.
- `torch.optim.SGD(...)` creates an optimizer.
- `lr` is the learning rate.
- `loss.backward()` computes gradients.
- `optimizer.step()` updates parameters.
- `torch.no_grad()` disables gradient tracking during inference.

## 9. Important Distinctions

| Operation | Purpose |
|---|---|
| `model(X)` | Produces predictions |
| `loss_fn(predictions, y)` | Measures prediction error |
| `optimizer.zero_grad()` | Clears accumulated gradients |
| `loss.backward()` | Computes gradients |
| `optimizer.step()` | Updates parameters |
| `model.parameters()` | Iterates over parameters |

A model can produce predictions before training. Those predictions may be poor because its parameters have not yet been learned from the data.

## 10. Practical Reminders

- The batch size and feature count are different concepts.
- The final layer's output size should match the task's desired output.
- Use a suitable loss function for the task.
- Training loss measures fit on the training data; validation and test data are needed to assess generalization.
- Reproducible experiments can use `torch.manual_seed(...)`, although reproducibility may depend on the hardware and operations used.

## Section Status

Section 10.3 — `torch.nn` and Model Building: **Practiced**

Next: Section 10.4 — Autograd and Gradient Inspection.