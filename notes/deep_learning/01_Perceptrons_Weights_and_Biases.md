# Perceptrons, Weights and Biases

## 1. What is a perceptron?

A perceptron is a simple artificial neuron used for binary classification. It receives inputs, combines them using weights and a bias, and produces a prediction.

## 2. Weighted sum

For one input:

`z = w*x + b`

For multiple inputs:

`z = w1*x1 + w2*x2 + ... + b`

- `x`: input feature
- `w`: weight associated with an input
- `b`: bias
- `z`: weighted sum, also called the score before activation

Example:

`x1 = 2, w1 = 3`
`x2 = 4, w2 = 0.5`
`b = -5`

`z = (3*2) + (0.5*4) - 5 = 3`

## 3. Weights

Weights control how much each input contributes to the score.

- A larger positive weight increases the score when its input is positive.
- A negative weight can decrease the score as its input increases.
- A weight near zero means the input has little influence in this model.

## 4. Bias

The bias shifts the score independently of the input values.

For example, if `z = wx + b`, changing `b` can change the prediction even when `x` and `w` remain unchanged.

## 5. Step activation

In our basic example:

- If `z >= 0`, predict `1`.
- If `z < 0`, predict `0`.

The activation rule turns the score into a class prediction.

## 6. How a perceptron learns

The model compares its prediction with the actual label:

`error = actual - predicted`

For the basic perceptron learning rule:

`w_new = w + learning_rate * error * x`

`b_new = b + learning_rate * error`

The learning rate controls the size of the update. In a multi-input perceptron, each weight is updated using its corresponding input.

A correct prediction has zero error, so the basic rule makes no update for that example.

## 7. Prediction versus training

A prediction function uses the current weights and bias to produce an output. It does not automatically learn.

Training repeatedly uses examples and their correct labels to adjust the parameters.

Modern neural networks generally use backpropagation with gradient-based optimizers rather than the basic perceptron update rule.

## 8. Python prediction example

```python
def perceptron_predict(x1, x2, w1, w2, bias):
    z = w1 * x1 + w2 * x2 + bias
    return 1 if z >= 0 else 0

prediction = perceptron_predict(2, 4, 3, 0.5, -5)
print("Prediction:", prediction)  # 1
```

## Key takeaway

A perceptron combines inputs using weights and bias, then applies an activation rule to make a prediction. Learning means adjusting those parameters using training examples.
