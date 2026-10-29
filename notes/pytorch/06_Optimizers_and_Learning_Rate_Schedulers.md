# PyTorch — Optimizers and Learning-Rate Schedulers

**File:** `machine_learning/notes/pytorch/06_Optimizers_and_Learning_Rate_Schedulers.md`  
**Roadmap:** Section 10.6 — PyTorch & Practical Deep Learning  
**Status:** Practiced

## 1. What is an optimizer?

An optimizer uses calculated gradients to update a model's trainable parameters, such as weights and biases.

Standard training sequence:

```python
optimizer.zero_grad()
predictions = model(X_batch)
loss = loss_fn(predictions, y_batch)
loss.backward()
optimizer.step()
```

- `zero_grad()` clears old gradients.
- `loss.backward()` computes gradients.
- `optimizer.step()` updates parameters.

## 2. Learning rate

The learning rate controls the size of optimizer updates.

For basic SGD:

`new_parameter = old_parameter - learning_rate * gradient`

A learning rate that is too small can slow training. One that is too large can cause unstable updates or prevent convergence.

## 3. Main optimizers

### SGD

Uses the current gradient to update parameters.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### SGD with Momentum

Uses a running direction informed by past gradients, which can reduce zigzagging and accelerate progress.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

### Adam

Tracks moving averages of gradients and squared gradients to adapt update sizes for individual parameters.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam is a common starting point for neural networks.

### AdamW

Uses Adam-style adaptive updates while handling weight decay separately from the gradient-based update.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

Weight decay can regularize a model, but excessive weight decay can make it underfit. Its value should be tuned for the task.

## 4. Learning-rate schedulers

A scheduler adjusts the optimizer's learning rate according to a rule.

### StepLR

Reduces the learning rate at fixed intervals.

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)
```

If called once per epoch, the learning rate is multiplied by 0.5 every 20 epochs.

### ExponentialLR

Multiplies the learning rate by a constant factor at each scheduler step.

### ReduceLROnPlateau

Reduces the learning rate when a monitored metric stops improving.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# After calculating validation loss:
scheduler.step(val_loss)
```

For `ReduceLROnPlateau`, pass the monitored metric. For many other schedulers, call `scheduler.step()` without a metric.

## 5. Practical: compare Adam and AdamW

**Coding file:** `machine_learning/10_pytorch/06_optimizers_and_schedulers.py`

The experiment uses a simple regression dataset with target relationship `y = 2x + 1`.

The model should learn a weight near 2 and a bias near 1. The practical compares Adam and AdamW and demonstrates a StepLR scheduler.

Observed results after adjusting the example:

| Metric | Adam | AdamW |
|---|---:|---:|
| Final loss | 0.0018 | 0.1251 |
| Weight | 1.9273 | 1.3938 |
| Bias | 1.0000 | 0.9935 |

Adam performed better in this particular run. AdamW's weight decay encouraged smaller weights and contributed to underfitting this simple target. This does not establish that Adam is generally superior to AdamW.

## 6. Common mistakes

- Assuming Adam is always better than AdamW.
- Assuming weight decay always improves performance.
- Confusing `scheduler.step()` with `optimizer.step()`.
- Assuming the scheduler calculates gradients.
- Using `ReduceLROnPlateau` without passing the monitored metric.
- Ignoring unstable losses or `NaN` values instead of checking scaling and hyperparameters.

## 7. Key takeaway

The optimizer determines how gradients update parameters. The scheduler changes the learning rate during training. Their effects depend on the model, dataset, and hyperparameters.

