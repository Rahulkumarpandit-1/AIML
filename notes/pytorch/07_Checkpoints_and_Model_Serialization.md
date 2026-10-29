# Checkpoints and Model Serialization

## 1. Why save checkpoints?

Checkpoints let us save a trained model and later use it for prediction or resume training without starting from scratch.

## 2. Model state

```python
torch.save(model.state_dict(), "model.pth")
```

- `model.state_dict()` contains learned parameters and persistent buffers.
- It does not contain the model architecture.
- Recreate a compatible model before loading the state.

## 3. Loading a model for inference

```python
model = MyModel().to(device)

state = torch.load(
    "model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state)
model.eval()

with torch.inference_mode():
    predictions = model(X)
```

`model.eval()` changes the behavior of layers such as Dropout and BatchNorm. `torch.inference_mode()` disables gradient tracking for inference.

## 4. Saving a training checkpoint

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": best_val_loss,
}

torch.save(checkpoint, CHECKPOINT_PATH)
```

A full training checkpoint can contain:

- Model state
- Optimizer state
- Last completed epoch
- Best validation loss
- Scheduler state, if a scheduler is used

## 5. Loading a training checkpoint

Create the model and optimizer first, then load the saved states.

```python
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
best_val_loss = checkpoint["best_val_loss"]
```

If a learning-rate scheduler is used, save and restore its state too.

## 6. Important differences

| Method | Purpose |
|---|---|
| `model.state_dict()` | Saves model parameters and persistent buffers |
| `optimizer.state_dict()` | Saves optimizer state, including Adam's internal estimates |
| `scheduler.state_dict()` | Saves learning-rate scheduler progress |
| `model.eval()` | Switches the model to evaluation behavior |
| `torch.inference_mode()` | Disables gradient tracking for inference |

## 7. Saving paths

```python
from pathlib import Path

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "training_checkpoint.pth"
)

torch.save(checkpoint, CHECKPOINT_PATH)
```

This saves the checkpoint beside the current Python script. The parent directory must already exist.

## 8. Key lessons

- Saving weights alone is usually sufficient for inference.
- Resuming training faithfully requires the optimizer state and training progress; restore scheduler state when applicable.
- `start_epoch = checkpoint["epoch"] + 1` prevents repeating the last completed epoch.
- Recreate the same compatible model architecture before loading.
- Use trusted checkpoint files and safe loading options such as `weights_only=True` when supported.
- A `.pth` extension is conventional; it does not itself determine the file contents.

## Status

**Section 10.7 — Practiced**

Practical completed: saved a checkpoint, restored model and optimizer states, checked restored parameters, and resumed training.