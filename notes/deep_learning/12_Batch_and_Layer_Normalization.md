# Batch Normalization and Layer Normalization

## 1. What is normalization?

Normalization transforms values into a more standardized scale. In neural networks, it can make optimization more stable and help training proceed effectively.

A common normalization formula is:

\[
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

Where:

- \(x\): original value
- \(\mu\): mean of the values being normalized
- \(\sigma^2\): variance of those values
- \(\epsilon\): small positive constant to prevent division by zero
- \(\hat{x}\): normalized value

After normalization, a learnable scale and shift may be applied:

\[
y=\gamma\hat{x}+\beta
\]

- \(\gamma\): learnable scale parameter
- \(\beta\): learnable shift parameter

These parameters let the network adjust the normalized values when needed.

**Important:** Normalization is not guaranteed to improve every model. Its benefits depend on the architecture, data, and training setup.

## 2. Batch Normalization (BatchNorm)

Batch Normalization normalizes each feature using statistics calculated across the examples in a mini-batch.

Consider this input:

| Example | Feature 1 | Feature 2 | Feature 3 |
|---|---:|---:|---:|
| A | 2 | 10 | 20 |
| B | 4 | 12 | 24 |
| C | 6 | 14 | 28 |

BatchNorm calculates statistics separately for each column:

- Feature 1 uses values 2, 4, 6.
- Feature 2 uses values 10, 12, 14.
- Feature 3 uses values 20, 24, 28.

It does not calculate one shared mean across all nine values.

### PyTorch implementation

For a 2D tensor shaped `[batch_size, features]`:

```python
import torch
import torch.nn as nn

batch_norm = nn.BatchNorm1d(num_features=3)

x = torch.tensor([
    [2.0, 10.0, 20.0],
    [4.0, 12.0, 24.0],
    [6.0, 14.0, 28.0]
])

output = batch_norm(x)
print(output)
```

`num_features=3` means the input has three features.

### Training and evaluation behavior

**During training:**
- BatchNorm normally calculates the current mini-batch's mean and variance.
- It updates running estimates of those statistics.

**During evaluation:**
- BatchNorm normally uses the running mean and variance accumulated during training.
- This behavior is controlled by the layer's training/evaluation mode.

```python
model.train()  # BatchNorm uses current batch statistics

model.eval()   # BatchNorm normally uses running statistics
```

BatchNorm maintains separate learnable scale and shift parameters for each feature by default.

### Common use cases

- Convolutional neural networks (CNNs)
- Some feedforward neural networks
- Architectures where batch-based normalization is appropriate

**Limitation:** Batch statistics can be unreliable with very small batches. In training mode, `BatchNorm1d` also generally requires more than one value per channel to calculate variance.

## 3. Layer Normalization (LayerNorm)

Layer Normalization calculates statistics across the features of each individual example.

Using the same input:

| Example | Feature 1 | Feature 2 | Feature 3 |
|---|---:|---:|---:|
| A | 2 | 10 | 20 |
| B | 4 | 12 | 24 |
| C | 6 | 14 | 28 |

LayerNorm works row by row:

- Example A uses 2, 10, 20.
- Example B uses 4, 12, 24.
- Example C uses 6, 14, 28.

It does not depend on the other examples in the batch.

### PyTorch implementation

```python
import torch
import torch.nn as nn

layer_norm = nn.LayerNorm(normalized_shape=3)

x = torch.tensor([
    [2.0, 10.0, 20.0],
    [4.0, 12.0, 24.0],
    [6.0, 14.0, 28.0]
])

output = layer_norm(x)
print(output)
```

`normalized_shape=3` means LayerNorm normalizes the final dimension containing three features.

### LayerNorm in transformers

Transformer models commonly represent tokens as feature vectors. For example, a tensor may have shape:

`[batch_size, sequence_length, 768]`

Each token representation has 768 features. This layer normalizes those features independently:

```python
layer_norm = nn.LayerNorm(768)
```

For each token, LayerNorm calculates the mean and variance across its 768 features. It does not need statistics from other examples in the batch.

### Training and evaluation behavior

LayerNorm calculates statistics from the current example's normalized dimensions in both training and evaluation modes. It does not normally maintain running mean and variance estimates.

This makes it naturally suitable for transformers and models where batch composition may vary.

## 4. BatchNorm vs LayerNorm

| Property | Batch Normalization | Layer Normalization |
|---|---|---|
| Statistics calculated across | Examples in a batch, separately per feature | Features within each example |
| Depends on other batch examples? | Yes, during training | No |
| Uses running statistics? | Yes, normally during evaluation | No, normally |
| Behavior in evaluation mode | Uses stored running statistics | Uses the example's own feature statistics |
| Common applications | CNNs and some feedforward networks | Transformers and LLMs |
| PyTorch example for 4 features | `nn.BatchNorm1d(4)` | `nn.LayerNorm(4)` |

**Memory trick:**
- **BatchNorm:** Think across the batch.
- **LayerNorm:** Think across the features of one example.

## 5. A direct comparison in PyTorch

```python
import torch
import torch.nn as nn

x = torch.tensor([
    [2.0, 4.0, 6.0, 8.0],
    [10.0, 12.0, 14.0, 16.0],
    [1.0, 3.0, 5.0, 7.0]
])

batch_norm = nn.BatchNorm1d(4)
layer_norm = nn.LayerNorm(4)

batch_norm.train()
layer_norm.train()

print("BatchNorm output:")
print(batch_norm(x))

print("\nLayerNorm output:")
print(layer_norm(x))
```

Interpretation:

- BatchNorm calculates statistics separately for each column.
- LayerNorm calculates statistics separately for each row.
- Both have learnable scale and shift parameters by default.
- The outputs need not be identical because the statistics are calculated over different dimensions.

For reproducible experiments, remember that BatchNorm's running statistics and learnable parameters are part of the layer's state.

## 6. Common mistakes

1. **Thinking both methods normalize the same values:** They use different dimensions to calculate statistics.
2. **Thinking `model.eval()` disables normalization:** It changes the behavior of certain layers; it does not switch normalization off.
3. **Confusing `torch.no_grad()` with evaluation mode:** `torch.no_grad()` disables gradient tracking. `model.eval()` changes the behavior of layers such as BatchNorm and Dropout.
4. **Using the wrong feature dimension:** `nn.LayerNorm(768)` expects the final dimension to be 768.
5. **Assuming BatchNorm is always better:** Small batches and the architecture can affect its suitability.
6. **Assuming normalization removes the need for other training practices:** Learning rate, initialization, data quality, and architecture still matter.

## 7. Quick revision

- Normalization uses a mean, variance, and a small epsilon.
- BatchNorm calculates statistics across examples, separately per feature.
- LayerNorm calculates statistics across features, separately per example.
- BatchNorm normally uses running statistics during evaluation.
- LayerNorm normally uses the current example's own feature statistics in both training and evaluation.
- BatchNorm is common in CNNs; LayerNorm is common in transformers and LLMs.
- In PyTorch, use `nn.BatchNorm1d(num_features)` for suitable 2D batch-feature inputs and `nn.LayerNorm(feature_count)` when normalizing the final feature dimension.

## 8. Self-test

1. Why does BatchNorm depend on batch statistics during training?
2. Which dimensions does LayerNorm normalize for an input shaped `[32, 768]` when using `nn.LayerNorm(768)`?
3. What statistics does BatchNorm normally use during evaluation?
4. Why is LayerNorm a natural choice for transformer feature vectors?
5. What is the difference between `model.eval()` and `torch.no_grad()`?

**Roadmap status:** Practiced — conceptual checkpoints completed; independent implementation remains to be tested.
