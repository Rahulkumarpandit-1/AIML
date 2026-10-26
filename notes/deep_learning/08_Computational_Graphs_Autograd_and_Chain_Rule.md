# Computational Graphs, Autograd, Chain Rule, and Gradient Descent

## 1. Computational Graphs

A computational graph represents the sequence of mathematical operations used to calculate an output.

Example:

`a = x * w`

`y = a + b`

For `x = 3`, `w = 2`, and `b = 1`:

- `a = 3 * 2 = 6`
- `y = 6 + 1 = 7`

The graph records how values depend on one another.

## 2. Forward Propagation vs. Backpropagation

**Forward propagation**
- Moves from inputs toward the output.
- Calculates intermediate values and predictions.

**Backpropagation**
- Works backward from the loss.
- Calculates gradients of the loss with respect to model parameters.
- Uses the chain rule to pass gradients through connected operations.

## 3. What Is a Gradient?

A gradient measures how sensitive an output or loss is to a variable.

For example:

`y = x^2`

The derivative is:

`dy/dx = 2*x`

When `x = 4`:
- Output: `y = 16`
- Gradient: `dy/dx = 8`

The output and gradient are different quantities.

## 4. Chain Rule

The chain rule calculates a gradient through connected operations.

Suppose:

`a = x^2`

`y = 3*a`

Then:

`dy/dx = (dy/da) * (da/dx)`

When `x = 2`:
- `a = 4`
- `y = 12`
- `dy/da = 3`
- `da/dx = 2*x = 4`

Therefore:

`dy/dx = 3 * 4 = 12`

The gradients of the connected operations are multiplied.

## 5. PyTorch Autograd

Autograd automatically calculates gradients through tracked operations.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

a = x ** 2
y = 3 * a

y.backward()

print("Output:", y.item())
print("Gradient:", x.grad.item())
```

Expected output:

```text
Output: 12.0
Gradient: 12.0
```

Important parts:
- `requires_grad=True`: tracks operations needed for gradient calculation.
- `y.backward()`: calculates gradients backward through the graph.
- `x.grad`: contains the gradient of `y` with respect to `x`.

`backward()` calculates gradients; it does not automatically update `x`.

## 6. Gradient Descent

Gradient descent uses gradients to update model parameters in a direction intended to reduce the loss.

Weight-update rule:

`new_weight = old_weight - learning_rate * gradient`

Example:
- Old weight = `5`
- Gradient = `2`
- Learning rate = `0.1`

`new_weight = 5 - (0.1 * 2) = 4.8`

If the gradient is negative, subtracting it can increase the weight.

## 7. Learning Rate

The learning rate controls the size of each gradient-descent update.

- **Too small:** learning may be slow.
- **Appropriate:** updates may make steady progress.
- **Too large:** updates may overshoot, oscillate, or diverge.

A large learning rate is not automatically better.

## 8. Full Training Cycle

`Forward propagation -> Calculate loss -> Backpropagation -> Optimizer update`

These steps repeat over training iterations.

- Forward propagation calculates predictions.
- The loss function measures prediction error.
- Backpropagation calculates gradients.
- The optimizer uses gradients to update weights and biases.

## Key Takeaways

1. Computational graphs represent dependencies between operations.
2. Forward propagation calculates values; backpropagation calculates gradients.
3. The chain rule connects gradients across multiple operations.
4. PyTorch autograd automates gradient calculation.
5. Gradient descent uses gradients to update parameters.
6. The learning rate controls update size.
## 9. Practical Training with PyTorch

### Goal

Train a single weight to learn the relationship:

\[
y = 3x
\]

Given input `x = 2` and target `y = 6`, the model starts with weight `w = 1`.

### Model and loss

The prediction is:

`prediction = w * x`

The squared-error loss is:

`loss = (prediction - target) ** 2`

Initially, the prediction is 2 and the loss is 16.

### Training steps

1. **Forward propagation:** calculate the prediction.
2. **Calculate loss:** measure the difference between prediction and target.
3. **Backpropagation:** use `loss.backward()` to calculate the gradient.
4. **Update weight:** subtract `learning_rate * gradient` from the current weight.
5. **Clear gradient:** use `w.grad.zero_()` before the next gradient calculation.

### Important PyTorch operations

- `requires_grad=True`: tells PyTorch to track operations needed to calculate gradients.
- `loss.backward()`: calculates gradients; it does not update the weight.
- `torch.no_grad()`: prevents gradient tracking during the manual parameter update.
- `w.grad.zero_()`: clears the accumulated gradient.

### What to observe

With learning rate `0.1`, the weight moves from `1.0` toward `3.0`, and the loss decreases.

The model is learning a parameter from an example; it has not automatically learned every possible relationship. A real neural network has many weights and usually uses a PyTorch optimizer to update them.