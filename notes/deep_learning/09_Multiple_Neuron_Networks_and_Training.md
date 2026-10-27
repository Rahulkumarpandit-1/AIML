# 09. Multiple-Neuron Networks and Training

## 1. What We Are Building

A neural network learns patterns from input data by adjusting its parameters (weights and biases).

A typical training program has six main stages:

1. Prepare training and validation data.
2. Create the model, loss function, and optimizer.
3. Train the model.
4. Evaluate it on validation data.
5. Use early stopping to decide when to stop.
6. Restore the best saved parameters.

**Overall flow:**

`Data → Model → Prediction → Loss → Gradients → Parameter Update → Validation → Save Best Model → Stop → Restore Best Model`

---

## 2. Training and Validation Data

We divide data into separate sets.

### Training data

- `X_train`: input features.
- `y_train`: correct target values.
- Used to calculate loss and update model parameters.

### Validation data

- `X_val`: validation inputs.
- `y_val`: correct validation targets.
- Used to evaluate how well the model performs on data not used for its training updates.

Example:

```python
X_train = torch.tensor([
    [1.0], [2.0], [3.0], [4.0], [5.0]
])

y_train = torch.tensor([
    [3.0], [5.0], [7.0], [9.0], [11.0]
])

X_val = torch.tensor([
    [6.0], [7.0], [8.0]
])

y_val = torch.tensor([
    [13.0], [15.0], [17.0]
])
```

The target pattern is `y = 2x + 1`.

---

## 3. Create the Model

```python
model = nn.Linear(1, 1)
```

This creates a linear layer with one input feature and one output.

Its prediction is:

`prediction = weight × input + bias`

The model starts with initialized parameters. During training, its weight and bias are updated to reduce prediction error.

## 4. Loss Function and Optimizer

```python
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

### Loss function

MSE means Mean Squared Error. It measures the average squared difference between predictions and correct target values.

Lower MSE generally indicates smaller prediction errors.

### Optimizer

SGD means Stochastic Gradient Descent.

- `model.parameters()` provides the parameters that the optimizer can update.
- `lr=0.01` is the learning rate, which controls the size of parameter updates.
- The optimizer uses calculated gradients to adjust weights and biases.

**Important distinction:**

- `loss.backward()` calculates gradients.
- `optimizer.step()` updates parameters using those gradients.

---

## 5. One Training Epoch

The training section:

```python
model.train()

prediction = model(X_train)
loss = loss_fn(prediction, y_train)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Execution order:

1. `model.train()` sets the model to training mode.
2. `model(X_train)` performs forward propagation and generates predictions.
3. `loss_fn(prediction, y_train)` calculates prediction error.
4. `optimizer.zero_grad()` clears gradients from the previous training step.
5. `loss.backward()` calculates gradients through backpropagation.
6. `optimizer.step()` updates the weights and bias.

A training epoch completes one pass through the training data used in this code.

---

## 6. Validation

```python
model.eval()

with torch.no_grad():
    val_prediction = model(X_val)
    val_loss = loss_fn(val_prediction, y_val)

current_val_loss = val_loss.item()
```

Validation evaluates the model after its training update.

- `model.eval()` switches the model to evaluation mode.
- `torch.no_grad()` disables gradient tracking.
- `model(X_val)` generates validation predictions.
- `loss_fn(...)` measures validation error.
- `.item()` extracts the numeric loss value from the PyTorch tensor.

**Validation measures performance; it does not update the weights in this code.**

---

## 7. Early Stopping

Early stopping prevents unnecessary training when validation performance stops improving.

### Initialize the tracking variables

```python
best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())

patience = 5
epochs_without_improvement = 0
max_epochs = 100
```

- `best_val_loss`: lowest validation loss recorded so far.
- `best_state`: independent copy of the model parameters associated with the best result.
- `patience`: number of consecutive non-improving epochs allowed.
- `epochs_without_improvement`: current count of consecutive failures to improve.
- `max_epochs`: maximum number of training epochs.

### Compare validation loss

```python
if current_val_loss < best_val_loss:
    best_val_loss = current_val_loss
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
```

If validation loss improves:

1. Update the best loss.
2. Save a copy of the current parameters.
3. Reset the counter to zero.

Otherwise, increase the counter by one.

### Example

| Epoch | Validation loss | Best loss | Counter |
|---|---:|---:|---:|
| 1 | 0.60 | 0.60 | 0 |
| 2 | 0.40 | 0.40 | 0 |
| 3 | 0.45 | 0.40 | 1 |
| 4 | 0.50 | 0.40 | 2 |
| 5 | 0.35 | 0.35 | 0 |

Epoch 5 improves the best loss, so the counter resets.

### Check the stopping condition

```python
if epochs_without_improvement >= patience:
    print(f"\nEarly stopping at epoch {epoch + 1}")
    break
```

When the counter reaches the patience limit, `break` exits the training loop.

An improvement resets the counter; it does not trigger early stopping.

---

## 8. Restore the Best Model

After the loop:

```python
model.load_state_dict(best_state)
```

Training may finish with parameters that are worse than those from an earlier epoch.

This line restores the saved best parameters.

For example:

- Epoch 8 achieves the lowest validation loss.
- Training continues until epoch 13.
- Validation loss does not improve for five consecutive epochs.
- `break` exits the loop.
- `load_state_dict(best_state)` restores the parameters from epoch 8.

The model's final parameters therefore correspond to the best saved validation result, not necessarily the last epoch.

---

## 9. Complete Program Flow

```text
Prepare training and validation data
              |
              v
Create model, loss function, optimizer
              |
              v
Initialize early-stopping variables
              |
              v
        Start an epoch
              |
              v
          Train model
              |
              v
       Calculate validation loss
              |
              v
     Did validation loss improve?
          /           \
        Yes            No
         |              |
  Save best state   Increase counter
  Reset counter          |
          \             /
           v           v
       Is patience exhausted?
          /           \
        No             Yes
        |               |
   Next epoch          break
                        |
                        v
              Restore best_state
                        |
                        v
                  Print results
```

Setup runs once. The training and validation sections repeat for each epoch. Restoration and final printing run after the loop ends.

---

## 10. Important Variables at a Glance

| Variable | Purpose |
|---|---|
| `model` | Current neural network |
| `prediction` | Predictions on training data |
| `loss` | Training error |
| `optimizer` | Updates model parameters |
| `val_prediction` | Predictions on validation data |
| `current_val_loss` | Current validation error as a Python number |
| `best_val_loss` | Lowest validation error recorded |
| `best_state` | Saved best model parameters |
| `epochs_without_improvement` | Consecutive non-improving epochs |
| `patience` | Allowed consecutive failures |
| `max_epochs` | Maximum number of epochs |

## 11. Key Takeaways

- The model produces predictions; the loss function measures their error.
- Backpropagation calculates gradients, and the optimizer updates parameters.
- Training data is used for learning; validation data is used for evaluation.
- Validation loss is compared across epochs to track the best result.
- Early stopping terminates training after enough consecutive failures to improve.
- `break` exits the loop, not the entire program.
- `best_state` is a saved snapshot of model parameters.
- `model.load_state_dict(best_state)` restores the best saved parameters.

**Remember the core sequence:**

`Train → Validate → Compare → Save or Count → Stop if Necessary → Restore Best Model`
