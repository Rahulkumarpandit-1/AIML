# Forward Propagation

## 1. What is Forward Propagation?

Forward propagation is the process of passing input data through a neural network to calculate its prediction.

Each neuron calculates a weighted sum, adds a bias, and applies an activation function.

## 2. The Two Main Calculations

**Step 1: Weighted sum**

`z = w1*x1 + w2*x2 + ... + b`

Where:
- `x` = input features
- `w` = weights
- `b` = bias
- `z` = weighted sum before activation

**Step 2: Activation**

`a = f(z)`

Where:
- `f` = activation function
- `a` = output after activation

Remember: `z` is the value before activation; `a` is the value after activation.

## 3. Worked Example: First Neuron

Given:

- `x1 = 2`, `x2 = 3`
- `w1 = 0.5`, `w2 = 0.2`
- `b = -0.5`

Calculate the weighted sum:

`z = (0.5*2) + (0.2*3) - 0.5`

`z = 1 + 0.6 - 0.5 = 1.1`

Using ReLU:

`ReLU(z) = max(0, z)`

Therefore:

`a = ReLU(1.1) = 1.1`

## 4. Passing Output to the Next Neuron

Suppose the next neuron receives `a = 1.1`, with weight `w = 2` and bias `b = -1`.

`z2 = (2*1.1) - 1 = 1.2`

Applying ReLU:

`a2 = ReLU(1.2) = 1.2`

The second neuron receives the first neuron's activated output, not its original inputs.

## 5. Choosing the Output Activation

- **Regression:** usually a linear output for continuous predictions.
- **Binary classification:** sigmoid commonly produces a probability between 0 and 1.
- **Multiclass classification:** softmax commonly produces probabilities across classes.

For example, a sigmoid output of `0.82` represents an estimated 82% probability for the positive class. A separate threshold can turn this probability into a class prediction.

## 6. Python Example

```python
def relu(z):
    return max(0, z)

# First neuron
x1, x2 = 2, 3
w1, w2 = 0.5, 0.2
b1 = -0.5

z1 = w1 * x1 + w2 * x2 + b1
a1 = relu(z1)

# Second neuron
w3 = 2
b2 = -1

z2 = w3 * a1 + b2
a2 = relu(z2)

print("First neuron output:", a1)
print("Final output:", a2)
```

Expected output:

```text
First neuron output: 1.1
Final output: 1.2
```

## 7. Forward Propagation vs. Backpropagation

- **Forward propagation:** calculates predictions by moving through the network.
- **Loss function:** measures the prediction error.
- **Backpropagation:** calculates gradients of the loss with respect to parameters.
- **Optimizer:** uses gradients to update weights and biases.

Typical training sequence:

`Forward pass -> Loss -> Backpropagation -> Optimizer update`

## Key Takeaways

1. Forward propagation calculates the network's output.
2. Every neuron uses weights, bias, and an activation function.
3. The next layer receives the previous layer's activation outputs.
4. Output activation depends on the task.
5. Forward propagation does not itself calculate gradients or update weights.
