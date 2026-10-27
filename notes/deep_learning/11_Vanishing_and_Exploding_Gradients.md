# Vanishing and Exploding Gradients

## 1. Introduction

During neural network training, backpropagation calculates gradients that indicate how the loss changes with respect to the model's parameters.

The optimizer uses these gradients to update the weights and reduce the loss.

Two problems can occur in deep neural networks:

- **Vanishing gradients:** gradients become extremely small.
- **Exploding gradients:** gradients become excessively large.

Both can make training difficult or unstable.

## 2. Recap: Gradient Descent

A simple gradient descent update is:

\[
w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial L}{\partial w}
\]

Where:

- \(w\) = weight
- \(L\) = loss
- \(\eta\) = learning rate
- \(\frac{\partial L}{\partial w}\) = gradient

The gradient influences the size and direction of the weight update.

A very small gradient produces a tiny update. An excessively large gradient can produce an unstable update.

## 3. Vanishing Gradients

### Definition

Vanishing gradients occur when gradients become extremely small as they propagate backward through a neural network.

### Causes

During backpropagation, gradients are multiplied by derivatives from successive layers. If these factors are repeatedly small, the gradients can shrink rapidly.

Common contributors include:

- Sigmoid saturation
- Repeated multiplication by small derivatives
- Poorly scaled weight initialization
- Certain deep network architectures

### Example

Suppose each layer contributes a factor of 0.2 to the gradient.

Across five layers:

\[
0.2^5=0.00032
\]

Repeated multiplication causes the gradient to become small.

This is a simplified illustration; actual gradients depend on the network's weights, activations, and loss function.

### Effects

- Earlier layers receive tiny updates.
- Learning becomes slow or ineffective.
- The network may struggle to learn useful representations.

## 4. Why Sigmoid Can Contribute to Vanishing Gradients

The sigmoid activation function is:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Its derivative is:

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

The derivative has a maximum value of 0.25 and approaches zero when the input is very positive or very negative.

When many small derivatives are multiplied during backpropagation, gradients can shrink significantly.

This is why sigmoid saturation is a common contributor to vanishing gradients in deep networks.

## 5. Exploding Gradients

### Definition

Exploding gradients occur when gradients become excessively large during backpropagation.

### Causes

- Repeated multiplication by factors larger than one
- Poorly scaled weights
- Unstable training dynamics
- Excessively large parameter updates

### Example

Suppose each layer contributes a factor of 3.

Across five layers:

\[
3^5=243
\]

A gradient that starts at 1 could grow to 243 in this simplified example.

### Effects

- Very large weight updates
- Unstable loss values
- Training that fluctuates dramatically
- Numerical errors such as `inf` or `NaN`

## 6. Vanishing vs Exploding Gradients

| Feature | Vanishing gradients | Exploding gradients |
|---|---|---|
| Gradient magnitude | Extremely small | Excessively large |
| Weight updates | Tiny | Potentially huge |
| Training effect | Slow or ineffective learning | Unstable learning |
| Common indicator | Earlier layers learn slowly | Loss becomes unstable |
| Useful approaches | Initialization, activation and architecture choices | Gradient clipping, initialization and learning-rate control |

## 7. Methods to Reduce These Problems

### 7.1 Suitable weight initialization

Initialization helps control the scale of signals and gradients.

- **Xavier initialization:** commonly used with tanh.
- **He initialization:** commonly used with ReLU.

These methods can reduce the risk of gradient problems but cannot guarantee their elimination.

### 7.2 Activation function choices

ReLU is:

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

For positive inputs, its derivative is 1. This can help gradients flow through active ReLU units.

However, ReLU can produce zero gradients for negative inputs, so it does not solve every vanishing-gradient problem.

### 7.3 Gradient clipping

Gradient clipping limits gradient magnitude before the optimizer updates the weights.

It is particularly useful for controlling exploding gradients.

In PyTorch:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

If the combined gradient norm exceeds the chosen maximum, the gradients are scaled down.

Gradient clipping does not directly fix vanishing gradients.

### 7.4 Learning-rate control

A learning rate that is too large can cause unstable parameter updates.

Reducing it may improve training stability, although it does not directly solve every exploding-gradient problem.

### 7.5 Normalization and architecture design

Batch normalization, layer normalization, and suitable network architectures can improve training dynamics in appropriate settings.

Their benefits depend on the model and the cause of the problem.

## 8. Gradient Clipping in a Training Loop

```python
import torch
import torch.nn as nn

model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 4.0]
])

y = torch.tensor([
    [1.0],
    [2.0]
])

optimizer.zero_grad()

prediction = model(X)
loss = loss_fn(prediction, y)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

### Code flow

1. `optimizer.zero_grad()` clears gradients from the previous iteration.
2. `model(X)` performs forward propagation.
3. `loss_fn()` calculates the loss.
4. `loss.backward()` calculates gradients.
5. `clip_grad_norm_()` limits the combined gradient norm when needed.
6. `optimizer.step()` updates the model parameters using the resulting gradients.

The order matters: clipping must happen after gradients are calculated and before the optimizer uses them.

## 9. Important Distinction

**Gradient magnitude and learning rate are not the same thing.**

- The gradient comes from differentiating the loss with respect to a parameter.
- The learning rate controls how strongly the optimizer applies the gradient.
