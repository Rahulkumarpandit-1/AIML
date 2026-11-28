# 18. Time-Series Forecasting

## 1. What is Time-Series Data?

Time-series data consists of observations recorded over time in chronological order.

Examples:
- Daily sales of a shop
- Monthly electricity consumption
- Hourly temperature readings
- Daily stock or commodity prices
- Website traffic over time

The order of observations matters because earlier values can help us understand and predict later values.

## 2. What is Time-Series Forecasting?

Time-series forecasting uses historical observations to estimate future values.

For example:

| Day | Sales |
|---|---:|
| Monday | 100 |
| Tuesday | 120 |
| Wednesday | 110 |
| Thursday | 140 |

Using previous sales observations, a model can estimate Friday's sales.

A forecast is an estimate, not a guarantee.

## 3. Why Does Order Matter?

In ordinary machine learning, randomly splitting data into training and testing sets is common.

For time-series forecasting, random splitting can cause future observations to appear in the training set while earlier observations appear in the test set.

This creates an unrealistic evaluation because, in a real forecasting situation, we should only use past information to predict the future.

Therefore, we generally split time-series data chronologically:

- Earlier observations → Training data
- Later observations → Testing data

This helps prevent future-data leakage.

## 4. Sliding-Window Technique

A sliding window converts a time series into input-target pairs that a supervised learning model can learn from.

Suppose:

```python
sales = [100, 120, 110, 140, 160, 150, 180]
window_size = 3
```

We use three previous observations to predict the next observation.

The resulting examples are:

| Input sequence | Target |
|---|---:|
| `[100, 120, 110]` | 140 |
| `[120, 110, 140]` | 160 |
| `[110, 140, 160]` | 150 |
| `[140, 160, 150]` | 180 |

The window moves forward by one observation at a time.

### Creating sliding windows

```python
X = []
y = []

for i in range(len(sales) - window_size):
    X.append(sales[i:i + window_size])
    y.append(sales[i + window_size])
```

- `X` contains the input sequences.
- `y` contains the next observation for each sequence.
- `sales[i:i + window_size]` selects the input window.
- `sales[i + window_size]` selects the next value as the target.

The number of generated examples is:

`Number of observations - Window size`

## 5. Why Use GRU for Forecasting?

A GRU (Gated Recurrent Unit) is a recurrent neural network architecture designed to process sequential information.

It processes observations in order and maintains a hidden state that carries information from earlier time steps.

For forecasting:

1. The model receives a historical sequence.
2. The GRU processes its observations in order.
3. Its hidden state represents information learned from the sequence.
4. The final hidden state is passed to a linear layer.
5. The linear layer produces a numerical prediction.

A GRU can learn patterns in sequential data, although its forecasting performance depends on the data, training process, and model design.

## 6. Preparing Time-Series Data in PyTorch

### Step 1: Define the data

```python
values = [10, 20, 30, 40, 50, 60]
window_size = 3
```

### Step 2: Generate input-target pairs

```python
X = []
y = []

for i in range(len(values) - window_size):
    X.append(values[i:i + window_size])
    y.append(values[i + window_size])
```

Result:

```text
Inputs:
[[10, 20, 30],
 [20, 30, 40],
 [30, 40, 50]]

Targets:
[40, 50, 60]
```

### Step 3: Convert to PyTorch tensors

```python
import torch

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

Neural networks work with tensors, so the Python lists are converted into PyTorch tensors.

### Step 4: Add the feature dimension

```python
X = X.unsqueeze(-1)
```

The resulting input shape is:

`(3, 3, 1)`

A GRU using `batch_first=True` expects:

`(batch_size, sequence_length, input_size)`

For this example:

| Dimension | Meaning | Value |
|---|---|---:|
| Batch size | Number of sequences | 3 |
| Sequence length | Time steps per sequence | 3 |
| Input size | Features per time step | 1 |

We use `unsqueeze(-1)` because each time step contains one feature: the sales value.

### Step 5: Split chronologically

```python
X_train = X[:2]
X_test = X[2:]

y_train = y[:2]
y_test = y[2:]
```

Training examples:

```text
[10, 20, 30] → 40
[20, 30, 40] → 50
```

Testing example:

```text
[30, 40, 50] → 60
```

The training data comes before the testing data.

### Step 6: Reshape target tensors

```python
y_train = y_train.unsqueeze(-1)
y_test = y_test.unsqueeze(-1)
```

Now:

- `y_train` has shape `(2, 1)`.
- `y_test` has shape `(1, 1)`.

These shapes match the model's expected prediction shape.

## 7. Building the GRU Forecasting Model

```python
import torch
from torch import nn

class GRUForecaster(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        output, hidden = self.gru(x)

        final_hidden = hidden[-1]

        prediction = self.fc(final_hidden)

        return prediction
```

### Important components

**`nn.Module`**

The base class for creating custom PyTorch neural network models.

**`super().__init__()`**

Initializes the parent `nn.Module` class so PyTorch can manage the model's layers and parameters.

**`nn.GRU()`**

Creates the GRU layer.

- `input_size`: Number of features at each time step.
- `hidden_size`: Number of values in the GRU hidden representation.
- `batch_first=True`: Input shape follows `(batch, sequence, features)`.

**`nn.Linear(hidden_size, 1)`**

Maps the GRU's hidden representation to one numerical output.

**`output, hidden = self.gru(x)`**

The GRU returns:
- `output`: Hidden representations for all time steps.
- `hidden`: Final hidden state for each GRU layer and direction.

**`hidden[-1]`**

Selects the final hidden state of the last GRU layer for this unidirectional model.

**`self.fc(final_hidden)`**

Converts the final hidden representation into a forecast.

## 8. Instantiate the Model

```python
model = GRUForecaster(
    input_size=1,
    hidden_size=16
)
```

- `input_size=1`: One sales value per time step.
- `hidden_size=16`: The GRU maintains a 16-dimensional hidden representation.

The hidden size is a model-design choice, not the number of time steps.

## 9. Loss Function and Optimizer

### Mean Squared Error

```python
loss_fn = nn.MSELoss()
```

MSE measures the average squared difference between predicted and actual numerical values.

For one prediction:

`Actual = 40`

`Predicted = 35`

Squared error:

`(40 - 35)² = 25`

MSE is commonly used for regression problems involving continuous numerical predictions.

### Adam Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

- `model.parameters()`: The trainable weights and biases.
- `Adam`: An optimization algorithm that updates the parameters using gradients.
- `lr`: Learning rate, which controls the size of parameter updates.

The learning rate is a hyperparameter and may need tuning.

## 10. Training the Model

```python
epochs = 200

for epoch in range(epochs):

    model.train()

    predictions = model(X_train)

    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {loss.item():.4f}"
        )
```

### Training steps

1. `model.train()` puts the model into training mode.
2. `model(X_train)` generates predictions.
3. `loss_fn()` measures the prediction error.
4. `optimizer.zero_grad()` clears gradients from the previous iteration.
5. `loss.backward()` calculates gradients through backpropagation.
6. `optimizer.step()` updates the trainable parameters.

An epoch means one complete pass through the training data.

The loop repeats this process for the specified number of epochs.

Because this is a tiny dataset, we process the entire training set in one batch. A DataLoader is not necessary for this demonstration.

## 11. Evaluating the Model

```python
model.eval()

with torch.no_grad():
    test_predictions = model(X_test)

print("Predicted:", test_predictions)
print("Actual:", y_test)

test_loss = loss_fn(test_predictions, y_test)

print("Test MSE:", test_loss.item())
```

**`model.eval()`**

Switches the model into evaluation mode.

**`torch.no_grad()`**

Disables gradient calculation because we are not updating parameters during evaluation.

**Training loss**

Measures error on examples used to train the model.

**Test loss**

Measures error on examples that were held out during training.

The test set should not be repeatedly used to tune the model.

## 12. Understanding Overfitting

Suppose:

- Training loss = 2.5
- Test loss = 40

This suggests that the model performs much better on training examples than on unseen examples.

This pattern can indicate overfitting.

Overfitting occurs when a model learns the training data too closely and fails to generalize well to unseen data.

However, our demonstration has only two training examples and one test example. That is not enough data to make a reliable diagnosis of overfitting.

## 13. Making a New Forecast

Suppose the latest observations are:

`[40, 50, 60]`

We want to predict the next observation.

### Prepare the input

```python
new_sequence = torch.tensor(
    [[[40.0], [50.0], [60.0]]],
    dtype=torch.float32
)
```

Its shape is `(1, 3, 1)`:

- 1 sequence.
- 3 time steps.
- 1 feature per time step.

### Generate the forecast

```python
model.eval()

with torch.no_grad():
    forecast = model(new_sequence)

print("Next predicted value:", forecast.item())
```

`forecast.item()` extracts the single numerical prediction from the tensor.

This is one-step-ahead forecasting: using a historical window to predict the next value.

The model does not automatically guarantee an accurate prediction. A meaningful forecasting system requires sufficient historical data, appropriate preprocessing, validation, and careful evaluation.

## 14. Important Concepts to Remember

| Concept | Meaning |
|---|---|
| Time series | Observations recorded over time |
| Forecasting | Estimating future values |
| Sliding window | Creating input-target sequences |
| Window size | Number of past observations used |
| Chronological split | Training on earlier data and testing on later data |
| GRU | Recurrent network for sequential data |
| Hidden state | Representation carrying information through a sequence |
| MSE | Measures squared prediction error |
| Adam | Optimizer that updates model parameters |
| Epoch | One complete pass through training data |
| Overfitting | Poor generalization to unseen data |
| One-step forecasting | Predicting the next observation |

## 15. Practical Workflow Summary

```text
Historical Time-Series Data
          ↓
Create Sliding Windows
          ↓
Separate Inputs and Targets
          ↓
Convert to PyTorch Tensors
          ↓
Add Feature Dimension
          ↓
Chronological Train/Test Split
          ↓
Build GRU Forecasting Model
          ↓
Define MSE Loss and Adam Optimizer
          ↓
Train the Model
          ↓
Evaluate on Test Data
          ↓
Forecast a New Value
```

## 16. Limitations of This Demonstration

- The dataset contains only six observations.
- Only two examples are used for training.
- Only one example is used for testing.
- No normalization or scaling is applied.
- The increasing values are an illustrative sequence, not a realistic forecasting dataset.

This practical demonstrates the GRU forecasting workflow. It does not demonstrate a reliable real-world forecasting system.