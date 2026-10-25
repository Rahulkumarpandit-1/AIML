# Gradient Descent

## 1. What is Gradient Descent?

Gradient Descent is an optimization algorithm used to adjust a model's parameters to reduce its loss.

- **Loss:** Measures how wrong the model's predictions are.
- **Gradient:** Indicates how the loss changes when a parameter changes.
- **Learning rate:** Controls the size of each update.
- **Weight:** A trainable parameter of the model.

## 2. Update Rule

`new_weight = old_weight - learning_rate × gradient`

Mathematically:

\[
w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial L}{\partial w}
\]

Here:
- \(w\) = weight
- \(\eta\) = learning rate
- \(\frac{\partial L}{\partial w}\) = gradient of loss with respect to the weight

We subtract the gradient because its sign tells us which direction locally increases the loss; moving in the opposite direction can reduce it.

## 3. Positive and Negative Gradients

**Positive gradient:** Subtracting it decreases the weight.

Example: weight = 5, learning rate = 0.5, gradient = 2.

`new_weight = 5 - (0.5 × 2) = 4`

**Negative gradient:** Subtracting a negative number increases the weight.

Example: weight = 5, learning rate = 0.5, gradient = -2.

`new_weight = 5 - (0.5 × -2) = 6`

The goal is to update parameters in a direction that reduces the loss.

## 4. Learning Rate

The learning rate controls how large each update is.

- **Too small:** Training may progress very slowly.
- **Too large:** Updates may overshoot the minimum, making training unstable.
- **Appropriate:** Often allows steady progress toward lower loss.

The best learning rate depends on the model, data, and optimization problem.

## 5. Repeated Updates

Gradient descent follows this general loop:

1. Make predictions with the current parameters.
2. Calculate the loss.
3. Calculate gradients.
4. Update parameters using an optimizer.
5. Repeat for more training steps.

In neural networks, backpropagation calculates gradients, while the optimizer uses them to update parameters.

## 6. Python Demonstration

This example minimizes `loss(w) = (w - 3) ** 2`.

```python
w = 0.0
learning_rate = 0.1

for step in range(10):
    loss = (w - 3) ** 2
    gradient = 2 * (w - 3)

    w = w - learning_rate * gradient

    new_loss = (w - 3) ** 2
    print(f"Step {step + 1}: weight={w:.3f}, loss={new_loss:.3f}")
```

The weight moves toward 3, where the loss is zero. Each iteration recalculates the gradient using the current weight.

This is an illustrative mathematical example, not a neural network training implementation.

## 7. Important Distinctions

- **Loss function:** Measures prediction error.
- **Gradient:** Describes how loss changes with a parameter.
- **Backpropagation:** Calculates gradients in a neural network.
- **Optimizer:** Uses gradients to update parameters.
- **Learning rate:** Controls the size of those updates.

**Key takeaway:** Gradient descent repeatedly adjusts model parameters using gradient information to reduce loss.