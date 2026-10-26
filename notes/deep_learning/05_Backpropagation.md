# Backpropagation

## 1. What is Backpropagation?

Backpropagation is an algorithm used to calculate gradients of a neural network's loss with respect to its trainable parameters.

These gradients tell us how changing a weight or bias affects the loss.

**Purpose:** Determine how the model's parameters influence its prediction error.

## 2. The Training Cycle

1. **Forward pass:** Inputs travel through the network to produce predictions.
2. **Loss calculation:** Compare predictions with the actual targets.
3. **Backpropagation:** Calculate gradients by working backward through the network.
4. **Optimizer update:** Use the gradients to adjust weights and biases.
5. Repeat for subsequent training steps.

## 3. Backpropagation vs. Gradient Descent

- **Loss function:** Measures prediction error.
- **Backpropagation:** Calculates gradients of the loss.
- **Gradient descent:** Uses gradients to update parameters.
- **Learning rate:** Controls the size of each update.

Backpropagation and gradient descent are complementary, but they perform different jobs.

## 4. Why Work Backward?

A neural network contains a sequence of calculations across layers. The final loss depends on the output, which depends on earlier layer calculations and their parameters.

Backpropagation uses the chain rule to calculate how the final loss changes with each parameter, working backward through those calculations.

In deeper networks, this allows gradients to be calculated efficiently for many weights and biases.

## 5. Numerical Example

Suppose:

- Input: `x = 2`
- Weight: `w = 3`
- Actual value: `y = 10`
- Prediction: `y_pred = w * x`
- Loss: `L = (y_pred - y) ** 2`

The prediction is:

`y_pred = 3 * 2 = 6`

The loss is:

`L = (6 - 10) ** 2 = 16`

For this model, the gradient with respect to the weight is:

`dL/dw = 2 * (w*x - y) * x`

Substituting the values:

`dL/dw = 2 * (6 - 10) * 2 = -16`

With learning rate `0.1`:

`new_weight = 3 - (0.1 * -16) = 4.6`

The new prediction is `4.6 * 2 = 9.2`, and the new loss is:

`(9.2 - 10) ** 2 = 0.64`

The loss decreases from `16` to `0.64`.

This is a simple one-weight example. Real neural networks calculate gradients across many layers and parameters.

## 6. Key Takeaway

**Forward pass → loss → backpropagation → optimizer update.**

Backpropagation determines how parameters affect the loss. The optimizer uses that information to improve the model's parameters.