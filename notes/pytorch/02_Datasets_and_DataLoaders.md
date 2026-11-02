# PyTorch: Datasets and DataLoaders

## 1. Dataset

A Dataset stores or provides examples used to train a model.

- `X`: input features
- `y`: target values or labels
- `len(dataset)`: number of examples
- `dataset[index]`: retrieves one example

### TensorDataset

`TensorDataset` pairs tensors along their first dimension.

```python
import torch
from torch.utils.data import TensorDataset

X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[35.0], [42.0], [55.0]])

dataset = TensorDataset(X, y)

print(len(dataset))
print(dataset[0])
```

Python indexing starts at zero, so `dataset[0]` returns the first example and target.

## 2. DataLoader

A DataLoader groups Dataset examples into batches.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for inputs, targets in loader:
    print(inputs, targets)
```

Important parameters:

- `batch_size`: examples per batch.
- `shuffle=True`: randomizes example order, commonly during training.
- `shuffle=False`: preserves example order.
- `drop_last=False`: keeps the final incomplete batch; this is the default.
- `drop_last=True`: discards the final incomplete batch.

Example: 5 examples with `batch_size=2` produce batch sizes of 2, 2, and 1 when `drop_last=False`.

## 3. Creating a Custom Dataset

A custom Dataset usually defines three methods:

- `__init__()`: stores or prepares the data.
- `__len__()`: returns the number of examples.
- `__getitem__(index)`: returns the input and target for one index.

```python
import torch
from torch.utils.data import Dataset

class StudyDataset(Dataset):
    def __init__(self):
        self.hours = [1, 2, 3, 4, 5]
        self.scores = [35, 42, 55, 68, 80]

    def __len__(self):
        return len(self.hours)

    def __getitem__(self, index):
        x = torch.tensor(
            [self.hours[index]], dtype=torch.float32
        )
        y = torch.tensor(
            [self.scores[index]], dtype=torch.float32
        )
        return x, y
```

The DataLoader calls the Dataset to retrieve examples and combines them into batches.

## 4. How the Data Flows

1. The Dataset provides one input-target pair at a time.
2. The DataLoader collects pairs into batches.
3. The training loop receives a batch of inputs and targets.
4. The model uses the inputs to make predictions.
5. The loss compares predictions with targets.

```text
Dataset → DataLoader → Batch → Model → Loss
```

## 5. Important Distinction

- **Dataset:** defines how individual examples are accessed.
- **DataLoader:** organizes those examples into batches and controls their order.

## 6. Practical Reminders

- Keep inputs and targets aligned.
- Use `shuffle=True` for many training datasets.
- Validation and test loaders commonly use `shuffle=False`.
- The final batch can be smaller than `batch_size`.
- A Dataset/DataLoader does not train the model by itself; the training loop does that.

## Section Status

Section 10.2 — Datasets and DataLoaders: Practiced.

Next: Section 10.3 — `torch.nn` and Model Building.