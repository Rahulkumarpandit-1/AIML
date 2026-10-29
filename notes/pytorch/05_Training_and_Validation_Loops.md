# PyTorch — Training and Validation Loops

**File:** `machine_learning/notes/pytorch/05_Training_and_Validation_Loops.md`  
**Roadmap:** Section 10.5 — PyTorch & Practical Deep Learning  
**Status:** Learning until the practical runs successfully

## 1. Purpose

A training loop repeatedly updates model parameters using training data. A validation loop measures performance on separate data without updating those parameters.

- **Training:** Learn from the training dataset.
- **Validation:** Monitor generalization and help select the best model.
- **Testing:** Evaluate the selected model on held-out test data.

Do not use the test set to choose epochs or tune training decisions.

## 2. Standard training loop

```python
for X_batch, y_batch in train_loader:
    optimizer.zero_grad()

    predictions = model(X_batch)
    loss = loss_fn(predictions, y_batch)

    loss.backward()
    optimizer.step()
```

### What each line does

1. `optimizer.zero_grad()` clears gradients left from the previous batch.
2. `model(X_batch)` calculates predictions using the current parameters.
3. `loss_fn(predictions, y_batch)` measures prediction error.
4. `loss.backward()` computes gradients and stores them in parameters' `.grad` attributes.
5. `optimizer.step()` uses those gradients to update parameters.

**Important:** `loss.backward()` calculates gradients; it does not update the weights. The update happens in `optimizer.step()`.

## 3. Training mode and evaluation mode

```python
model.train()

# Training operations...

model.eval()
with torch.no_grad():
    # Validation operations...
```

- `model.train()` sets training behavior for modules such as Dropout and BatchNorm.
- `model.eval()` sets evaluation behavior for those modules.
- `torch.no_grad()` disables gradient tracking for operations inside its context.
- `model.eval()` and `torch.no_grad()` are different and often used together during validation.

Validation still calculates predictions and loss. It simply does not need to calculate gradients for ordinary evaluation.

## 4. Calculating average training loss

With `nn.MSELoss()` using its default `reduction="mean"`, each batch loss is a mean over that batch.

```python
total_train_loss = 0.0

for X_batch, y_batch in train_loader:
    # Training steps...
    total_train_loss += loss.item() * len(X_batch)

avg_train_loss = total_train_loss / len(train_loader.dataset)
```

Multiplying by the batch size accounts for the number of samples represented by each batch. Dividing by the total number of training samples produces a sample-weighted average.

This handles a smaller final batch correctly.

## 5. Training loss versus validation loss

- **Training loss:** How well the model fits data it learns from.
- **Validation loss:** How well it performs on separate data used for monitoring.

A model may begin overfitting when training loss keeps decreasing while validation loss starts increasing.

For example:

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.80 | 0.90 |
| 2 | 0.40 | 0.50 |
| 3 | 0.20 | 0.30 |
| 4 | 0.10 | 0.42 |
| 5 | 0.05 | 0.60 |

Epoch 3 has the lowest validation loss in this example. Later epochs fit the training data better but perform worse on validation data.

## 6. Saving the best checkpoint

```python
best_val_loss = float("inf")

if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    torch.save(model.state_dict(), "best_model.pth")
```

- `float("inf")` initializes the best loss to infinity.
- A lower validation loss indicates an improvement over the best observed value.
- `model.state_dict()` contains the model's parameters and persistent buffers.
- The file is overwritten whenever validation loss improves.

Load the saved parameters into the same model architecture:

```python
model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)
```

This restores the model parameters. It does not restore optimizer state, epoch number, or other training settings.

## 7. Early stopping

Early stopping ends training when validation performance fails to improve for a specified number of epochs.

```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    break
```

Early stopping does not automatically restore the best weights. Save checkpoints and load the best one afterward if needed.

## 8. Complete practical

**Coding file:** `machine_learning/10_pytorch/05_training_validation_loops.py`

Use a simple regression problem where `y = 2x + 1`.

The practical should demonstrate:

1. Creating training and validation datasets.
2. Loading training data in batches.
3. Training a model with forward propagation, loss, backward propagation, and optimizer updates.
4. Calculating average training loss.
5. Evaluating validation loss without gradient tracking.
6. Saving the model with the best validation loss.
7. Reloading the saved parameters.
8. Inspecting the learned weight and bias.

For this example, a trained model should learn parameters reasonably close to weight `2` and bias `1`. Exact values depend on initialization, training duration, and learning rate.

## 9. Common mistakes

- Forgetting `optimizer.zero_grad()`, causing gradients to accumulate unintentionally.
- Forgetting `loss.backward()`, so gradients are not computed for the current loss.
- Forgetting `optimizer.step()`, so parameters are not updated.
- Assuming `model.eval()` disables gradient tracking.
- Calculating a plain average of batch means when batch sizes differ.
- Selecting the best model using test-set performance.
- Assuming a model checkpoint automatically includes the optimizer state.

## 10. Key takeaway

The training loop learns by updating parameters. The validation loop measures performance without updating them. Validation results help select a good checkpoint, while the untouched test set provides the final evaluation.
