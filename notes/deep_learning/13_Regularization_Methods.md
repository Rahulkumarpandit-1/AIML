# Regularization Methods in Deep Learning

## 1. What is overfitting?

Overfitting occurs when a model learns the training data too specifically, including noise or accidental patterns, and performs poorly on unseen data.

Example:

| Model | Training accuracy | Test accuracy | Interpretation |
|---|---:|---:|---|
| A | 65% | 63% | Likely underfitting |
| B | 99% | 72% | Likely overfitting |
| C | 92% | 90% | Better generalization |

The goal is not simply to achieve high training accuracy. The model should also perform well on unseen data.

## 2. What is regularization?

Regularization refers to techniques that reduce overfitting and encourage better generalization.

The three methods covered here are:

1. Dropout
2. Weight decay
3. Early stopping

They work differently and can sometimes be used together.

---

## 3. Dropout

### Definition

Dropout randomly sets a proportion of activations to zero during training.

It helps prevent the network from becoming overly dependent on particular neurons or activation patterns.

### Example

Suppose a layer produces these activations:

`[0.8, 0.5, 0.9, 0.3]`

After a possible dropout mask, the activations might become:

`[0.8, 0.0, 0.9, 0.0]`

This is an illustrative example. The actual mask is randomly generated during training.

### Dropout probability

If the dropout probability is `p = 0.3`, approximately 30% of eligible activations are set to zero on average during each training pass.

It does not mean:
- 30% of the dataset is deleted.
- 30% of the model's weights are permanently removed.
- Model accuracy decreases by 30%.

### PyTorch implementation

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(32, 1)
)
```

Explanation:

- `nn.Linear(10, 32)`: transforms 10 input features into 32 outputs.
- `nn.ReLU()`: applies the ReLU activation.
- `nn.Dropout(p=0.3)`: randomly sets some activations to zero during training.
- `nn.Linear(32, 1)`: produces one output.

### Training and evaluation

```python
model.train()  # Dropout is active
model.eval()   # Dropout is disabled
```

Standard PyTorch dropout uses inverted dropout: surviving activations are scaled by `1 / (1 - p)` during training. During evaluation, dropout is disabled and this scaling is not applied.

### Important points

- Dropout changes activations, not the permanent structure of the model.
- The dropped activations can differ between training passes.
- Dropout is often useful in neural networks with overfitting problems.
- Excessive dropout can cause underfitting.

---

## 4. Weight Decay

### Definition

Weight decay discourages excessively large model weights during training.

Large weights can sometimes make a model overly sensitive to small changes in input data. Penalizing them can help improve generalization.

### L2 regularization objective

A common L2-regularized objective is:

\[
L_{\text{total}}=L_{\text{data}}+\lambda\sum_i w_i^2
\]

Where:

- \(L_{\text{data}}\): original task loss.
- \(w_i\): individual model weights.
- \(\lambda\): regularization strength.
- \(L_{\text{total}}\): objective being optimized.

A larger \(\lambda\) applies a stronger penalty. If it is too large, the model may underfit.

### PyTorch implementation

```python
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

Explanation:

- `lr=0.001`: learning rate.
- `weight_decay=0.01`: weight-decay strength.
- `AdamW`: applies decoupled weight decay alongside its adaptive gradient updates.

When using AdamW's `weight_decay` argument, you normally do not need to add a separate L2 penalty to the loss yourself.

Weight decay is commonly applied to weights, while biases and some normalization parameters may be excluded depending on the optimizer setup.

### Important points

- Weight decay influences parameter updates.
- It discourages excessively large weights; it does not force every weight to become zero.
- Strong weight decay can cause underfitting.
- The best value should be selected using validation data.

---

## 5. Early Stopping

### Definition

Early stopping monitors validation performance during training and stops when improvement has stalled for a specified number of checks.

It helps prevent unnecessary training after the model has reached its best observed validation performance.

### Example

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.80 | 0.85 |
| 2 | 0.55 | 0.60 |
| 3 | 0.38 | 0.43 |
| 4 | 0.25 | 0.35 |
| 5 | 0.17 | 0.39 |
| 6 | 0.12 | 0.48 |

The training loss keeps decreasing, but the validation loss reaches its minimum at Epoch 4 and then increases.

This suggests the model may be overfitting. Epoch 4 is the best checkpoint among these examples.

### Important terms

- **Validation loss:** Loss on data used to monitor training and tune model choices.
- **Best validation loss:** Lowest validation loss observed so far.
- **Patience:** Number of consecutive validation checks without improvement allowed before stopping.
- **Best checkpoint:** Saved model parameters corresponding to the best monitored validation performance.

### PyTorch implementation: core logic

```python
import torch

best_val_loss = float("inf")
patience = 3
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()

    # Run the training loop here.
    train_one_epoch(model, train_loader, optimizer, loss_fn)

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_val_loss += loss.item()

    val_loss = total_val_loss / len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered")
        break
```

This is core logic, not a complete runnable program. It assumes that `model`, `train_loader`, `val_loader`, `optimizer`, `loss_fn`, `num_epochs`, and `train_one_epoch()` are defined. For unequal batch sizes, calculate the validation loss using an appropriate sample-weighted average if necessary.

### Restoring the best checkpoint

After training, restore the best saved parameters:

```python
model.load_state_dict(
    torch.load("best_model.pt", weights_only=True)
)
model.eval()
```

This assumes the saved file contains a compatible model `state_dict` and your PyTorch version supports `weights_only=True`.

**Important:** Early stopping does not guarantee that training stops at the exact epoch where validation loss first increases. Patience allows temporary fluctuations.

---

## 6. Comparison of the Three Methods

| Property | Dropout | Weight decay | Early stopping |
|---|---|---|---|
| Main action | Randomly disables activations | Discourages large weights | Limits training duration |
| Applied during | Training | Optimizer updates | Monitors training progress |
| Uses validation data directly? | No | Used for tuning strength | Yes |
| Main benefit | Reduces dependence on specific activation patterns | Controls parameter magnitudes | Avoids unnecessary training |
| Main risk if too aggressive | Underfitting | Underfitting | Stopping too early |

### Memory trick

- **Dropout:** Randomly disable activations.
- **Weight decay:** Discourage large weights.
- **Early stopping:** Stop when validation improvement stalls.

---

## 7. Recommended Training Workflow

1. Split data into training, validation, and test sets.
2. Train the model using training data.
3. Monitor validation performance.
4. Tune dropout probability and weight-decay strength using validation results.
5. Use early stopping and save the best checkpoint.
6. Evaluate the selected model on the untouched test set.

Do not repeatedly tune hyperparameters against the test set. Doing so can make the test results overly optimistic.

Regularization is not a substitute for appropriate data preparation, model architecture, learning rate, or evaluation.

---

## 8. Self-Test

1. A model achieves 99% training accuracy and 70% test accuracy. What problem is likely occurring?
2. What does `nn.Dropout(p=0.3)` do during training?
3. Why can excessive dropout cause underfitting?
4. What is the purpose of `weight_decay` in AdamW?
5. What does `patience=3` normally mean in early stopping?
6. Why should we save the checkpoint with the best validation performance?
7. What is the difference between training loss and validation loss?

## 9. Roadmap Status

**Section:** 09 — Deep Learning Foundations  
**Topic:** Regularization Methods  
**Status:** Practiced

Conceptual checkpoints completed successfully. Independent implementation and evaluation on a real dataset are still needed before marking this topic Mastered.
