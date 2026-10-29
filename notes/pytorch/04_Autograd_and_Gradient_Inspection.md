# Autograd and Gradient Inspection

## 1. What is Autograd?

PyTorch Autograd automatically calculates derivatives (gradients) of tensor operations. These gradients are used to train neural networks through backpropagation.

## 2. Enable Gradient Tracking

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

y.backward()

print(x.grad.item())  # 6.0
```

- `requires_grad=True`: tracks operations for gradient calculation.
- `backward()`: calculates gradients.
- `x.grad`: stores the calculated gradient.

For `y = x²`, the derivative is `dy/dx = 2x`. At `x = 3`, the gradient is `6`.

## 3. Chain Rule

For `a = 2x + 1` and `y = a²`:

```python
x = torch.tensor(2.0, requires_grad=True)
a = 2 * x + 1
y = a ** 2
y.backward()

print(x.grad.item())  # 20.0
```

The chain rule multiplies gradients through connected operations:

`dy/dx = (dy/da) * (da/dx)`

## 4. Computation Graph

PyTorch tracks operations involving tensors that require gradients.

- `requires_grad`: whether gradient tracking is enabled.
- `grad_fn`: the tracked operation that produced a non-leaf tensor.
- `grad`: the stored gradient, usually populated after `backward()`.

Leaf tensors created directly by the user normally have `grad_fn = None`.

## 5. Gradient Accumulation

Gradients accumulate when `backward()` is called repeatedly.

```python
x = torch.tensor(2.0, requires_grad=True)

(3 * x).backward()
(4 * x).backward()

print(x.grad.item())  # 7.0
```

The gradients `3` and `4` are added. In training, clear gradients before the next backward pass:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 6. Disable Gradient Tracking

```python
with torch.no_grad():
    predictions = model(X)
```

This prevents gradient tracking inside the block. It is useful during inference when gradients are not needed.

## 7. Inspect Model Gradients

```python
predictions = model(X)
loss = loss_fn(predictions, y_true)

optimizer.zero_grad()
loss.backward()

for name, parameter in model.named_parameters():
    print(name, parameter.grad)
```

- `parameter.data`: current parameter values.
- `parameter.grad`: gradients calculated for parameters.
- `loss.backward()`: calculates gradients; it does not update weights.
- `optimizer.step()`: updates weights and biases.

## 8. Standard Training Sequence

1. Forward pass: calculate predictions.
2. Calculate loss: measure prediction error.
3. Clear old gradients: `optimizer.zero_grad()`.
4. Backward pass: `loss.backward()`.
5. Update parameters: `optimizer.step()`.

## Key Takeaway

Autograd calculates gradients; the optimizer uses them to update model parameters. Gradients accumulate by default, so clear them appropriately during training.
