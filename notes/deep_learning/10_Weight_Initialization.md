# Weight Initialization

## 1. What Is Weight Initialization?

Weight initialization is the process of assigning initial values to the weights of a neural network before training begins.

A neuron calculates:

\[
z = wx + b
\]

\[
\hat{y} = f(z)
\]

Where:
- `x` = input
- `w` = weight
- `b` = bias
- `f(z)` = activation function
- `ŷ` = predicted output

The initial weights influence the neuron's output. During training, backpropagation calculates gradients, and the optimizer updates the weights to reduce the loss.

## 2. Why Is Weight Initialization Important?

Good initialization helps a neural network begin training with suitable weight values.

Poor initialization can cause:
- The symmetry problem
- Vanishing gradients
- Exploding gradients
- Slow or unstable training

### 2.1 The Symmetry Problem

Suppose multiple neurons in the same hidden layer have identical weights and biases.

If they receive identical inputs, they produce identical outputs. Under the same training conditions, they also receive identical gradient updates.

Consequently, these neurons can continue learning the same features instead of developing different roles.

**Solution:** Initialize weights with appropriately scaled random values to help break symmetry.

Initializing every weight to zero is therefore generally unsuitable for ordinary hidden layers.

Note: zero biases are often acceptable. The main issue is initializing all neurons' weights identically.

## 3. Random Weight Initialization

Random initialization assigns different starting values to weights.

However, weights should not be arbitrarily large or small. Their scale affects signal propagation during forward propagation and gradient propagation during backpropagation.

Two common initialization methods are Xavier and He initialization.

## 4. Xavier Initialization

Xavier initialization is also called Glorot initialization.

It is commonly used with `tanh` activation and can also be useful with sigmoid activation.

Its goal is to maintain a reasonably stable variance of signals across layers.

### Uniform version

For a layer with `n_in` input connections and `n_out` output connections:

\[
W \sim U(-a,a)
\]

where

\[
a = \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
\]

Here, `U(-a, a)` means weights are sampled from a uniform distribution between `-a` and `a`.

The layer sizes determine the range of initial weights.

## 5. He Initialization

He initialization is also called Kaiming initialization.

It is commonly used with ReLU activation.

ReLU is defined as:

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

Since ReLU sets negative activations to zero, He initialization chooses a weight scale designed to help maintain signal variance.

### Normal version

\[
W \sim \mathcal{N}\left(0,\frac{2}{n_{\text{in}}}\right)
\]

Here:
- `N` represents a normal distribution.
- Mean = `0`.
- Variance = `2 / n_in`.
- Standard deviation = `sqrt(2 / n_in)`.

### Example

Suppose a layer has four input connections.

\[
\sigma = \sqrt{\frac{2}{4}}
\]

\[
\sigma \approx 0.707
\]

If the layer has sixteen input connections:

\[
\sigma = \sqrt{\frac{2}{16}}
\approx 0.354
\]

The standard deviation decreases as the number of input connections increases.

## 6. Xavier vs He Initialization

| Feature | Xavier | He |
|---|---|---|
| Alternative name | Glorot | Kaiming |
| Common activation | Tanh | ReLU |
| Main objective | Maintain suitable signal variance | Account for ReLU's effect on signal variance |
| Scaling depends on | Input and output layer sizes | Primarily input layer size in the common normal version |

These are common starting choices, not universal rules for every architecture or activation function.

## 7. Weight Initialization in PyTorch

### 7.1 Create a neural network

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)
```

Network structure:

- Input: 4 features
- First linear layer: 4 inputs → 8 outputs
- ReLU activation
- Second linear layer: 8 inputs → 2 outputs

PyTorch automatically initializes the weights when the layers are created.

### 7.2 Apply He initialization

```python
nn.init.kaiming_normal_(
    model[0].weight,
    nonlinearity="relu"
)
```

Explanation:

- `model[0]` selects the first layer.
- `.weight` accesses its weight tensor.
- `kaiming_normal_()` applies He/Kaiming normal initialization.
- `nonlinearity="relu"` specifies the intended activation.
- The trailing underscore indicates an in-place operation.

To initialize the second layer's weights explicitly:

```python
nn.init.kaiming_normal_(
    model[2].weight,
    nonlinearity="relu"
)
```

**Important:** Choose initialization based on the architecture and activation functions. Do not blindly apply the same method to every layer. The example illustrates the API; output layers may require a different choice.

## 8. Vanishing and Exploding Gradients

Poorly scaled weights can contribute to gradient problems in deep networks.

### Vanishing gradients

Gradients become extremely small as they propagate backward.

Effects:
- Earlier layers receive tiny updates.
- Learning can become very slow.
- Some layers may learn ineffectively.

### Exploding gradients

Gradients become excessively large.

Effects:
- Parameter updates can become unstable.
- Loss may fluctuate dramatically.
- Training may produce numerical errors.

Appropriate initialization helps reduce these risks but does not guarantee that they will never occur.

## 9. Important Distinction

Weight initialization and weight optimization are different operations.

1. **Initialization:** Assigns the starting weight values.
2. **Forward propagation:** Calculates predictions.
3. **Loss calculation:** Measures prediction error.
4. **Backpropagation:** Calculates gradients.
5. **Optimizer step:** Updates the weights using those gradients.

Initialization happens before normal training updates begin.

## 10. Quick Revision

- Weight initialization sets initial weights before training.
- Identical initial weights can cause neurons to remain symmetric.
- Random initialization helps neurons develop different roles.
- Xavier initialization is commonly associated with `tanh`.
- He initialization is commonly associated with ReLU.
- Vanishing gradients become too small.
- Exploding gradients become too large.
- PyTorch provides `nn.init.xavier_uniform_()` and `nn.init.kaiming_normal_()` for common initialization strategies.
- Initialization helps training but does not replace backpropagation or optimization.

## 11. Self-Check Questions

1. What is weight initialization?
2. Why is initializing all hidden-layer weights to zero problematic?
3. What is the symmetry problem?
4. Which initialization method is commonly used with ReLU?
5. Which initialization method is commonly used with tanh?
6. What is the difference between variance and standard deviation?
7. What does `model[0].weight` access in PyTorch?
8. How does vanishing gradients differ from exploding gradients?
9. What is the difference between weight initialization and an optimizer update?

