# Activation Functions

## 1. What is an activation function?

An activation function transforms the weighted sum calculated by a neuron into its output.

First, the neuron calculates:

`z = w1*x1 + w2*x2 + ... + b`

Then the activation function calculates:

`output = f(z)`

Activation functions determine how a neuron responds to its inputs.

## 2. Why do we need activation functions?

Without nonlinear activation functions, stacking multiple neural-network layers would still result in a linear transformation. The network would not gain the ability to learn more complex nonlinear relationships simply by adding layers.

Nonlinear activations help neural networks learn complicated patterns in images, text, audio, and other data.

## 3. Step function

Definition:

- If `z >= 0`, output `1`.
- If `z < 0`, output `0`.

Examples:

- `step(-3) = 0`
- `step(0) = 1`
- `step(4) = 1`

**Use:** Understanding basic perceptrons and binary decisions.

**Limitation:** Its discontinuity and zero gradient almost everywhere make it unsuitable for standard gradient-based neural-network training.

## 4. Sigmoid

Formula:

`sigmoid(z) = 1 / (1 + exp(-z))`

Output range: `(0, 1)`.

Behavior:

- Large negative input → output near `0`
- Zero input → output `0.5`
- Large positive input → output near `1`

**Use:** Commonly used in the output layer for binary classification.

**Limitation:** Its gradients become very small for extreme inputs, which can slow learning in deeper networks.

A sigmoid output is a probability-like score; it is not automatically guaranteed to be well calibrated.

## 5. Tanh

Formula:

`tanh(z)`

Output range: `(-1, 1)`.

Behavior:

- Large negative input → output near `-1`
- Zero input → output `0`
- Large positive input → output near `1`

**Use:** Some recurrent neural networks and specialized hidden layers.

**Advantage:** Its outputs are centered around zero.

**Limitation:** It can also suffer from vanishing gradients for extreme inputs.

## 6. ReLU

Formula:

`ReLU(z) = max(0, z)`

Behavior:

- `ReLU(-3) = 0`
- `ReLU(0) = 0`
- `ReLU(4) = 4`

**Use:** A common default for hidden layers in many neural networks.

**Advantage:** Simple and computationally efficient; positive inputs pass through unchanged.

**Limitation:** Some neurons can become inactive and keep producing zero outputs, sometimes called the dying ReLU problem.

## 7. Quick comparison

| Function | Output range | Common use |
|---|---|---|
| Step | `0` or `1` | Basic perceptron concepts |
| Sigmoid | `(0, 1)` | Binary classification output |
| Tanh | `(-1, 1)` | Some recurrent networks |
| ReLU | `[0, infinity)` | Hidden layers |

These are common patterns, not absolute rules. The best activation depends on the architecture and task.

## 8. Python implementation

```python
import math

def step(z):
    return 1 if z >= 0 else 0

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def tanh(z):
    return math.tanh(z)

def relu(z):
    return max(0, z)

z = -3

print("Step:", step(z))
print("Sigmoid:", round(sigmoid(z), 3))
print("Tanh:", round(tanh(z), 3))
print("ReLU:", relu(z))
```

Expected output:

```text
Step: 0
Sigmoid: 0.047
Tanh: -0.995
ReLU: 0
```

## 9. Key takeaways

1. A neuron calculates a weighted sum before applying its activation.
2. Nonlinear activations allow neural networks to learn nonlinear patterns.
3. Sigmoid is commonly used for binary classification outputs.
4. ReLU is a common default for hidden layers.
5. Tanh produces zero-centered outputs between `-1` and `1`.
6. The step function helps explain perceptrons but is not appropriate for ordinary gradient-based training.
