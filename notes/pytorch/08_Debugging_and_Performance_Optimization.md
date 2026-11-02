# 08 — Debugging and Performance Optimization

## 1. Why Debugging Matters

A model can run without errors and still learn incorrectly. Debugging helps identify invalid data, incorrect tensor shapes, exploding gradients, and training problems.

## 2. Check Tensor Shapes and Devices

```python
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X device:", X.device)
print("Model device:", next(model.parameters()).device)
```

Input tensors and model parameters must be on compatible devices. For regression, predictions and targets should have matching shapes.

## 3. Detect NaN and Infinite Values

```python
print(torch.isfinite(X).all())
print(torch.isfinite(y).all())
print(torch.isfinite(loss))
```

- `NaN`: Not a Number.
- `inf`: Infinite value.
- `torch.isfinite()`: Identifies finite values.

If loss becomes NaN, inspect inputs, targets, loss calculations, gradients, and the learning rate.

## 4. Inspect Gradients

```python
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm())
        print("Finite:", torch.isfinite(param.grad).all())
```

Gradients indicate how parameters should change during optimization. Missing, non-finite, or unusually large gradients can signal problems.

## 5. Gradient Clipping

Gradient clipping limits gradient magnitude and can help prevent exploding gradients.

```python
optimizer.zero_grad()

prediction = model(X)
loss = loss_fn(prediction, y)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

Correct order: calculate loss → backpropagate → clip gradients → update parameters.

## 6. Avoid Unnecessary GPU Synchronization

Calling `.item()` frequently on a CUDA tensor can force the CPU to wait for GPU work to finish.

Instead of extracting a Python number every batch, accumulate loss tensors and extract the result less frequently.

## 7. DataLoader and Batching

```python
dataset = TensorDataset(X.cpu(), y.cpu())

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)
```

Move each batch to the model's device inside the loop.

For 100 examples with a batch size of 16, one complete pass contains 7 batches.

One pass is one epoch. More epochs mean repeating the dataset.

## 8. Profiling

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True
) as prof:
    # Run the operations you want to measure.
    ...

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10
    )
)
```

The profiler records operation timing and call counts.

Important columns:
- `Name`: operation name.
- `Self CPU`: CPU time attributed directly to an operation.
- `Self CUDA`: GPU time attributed directly to an operation.
- `# of Calls`: number of recorded calls.

`aten::addmm` commonly appears for linear-layer matrix multiplication and addition.

Profiling has overhead. Tiny models are not reliable benchmarks of overall GPU performance.

## 9. Practical Lessons

- Check shapes and devices before training.
- Check data, loss, and gradients when training becomes unstable.
- Clip gradients when appropriate.
- Use batches for larger datasets.
- Avoid unnecessary CPU/GPU synchronization.
- Profile before optimizing.
- Evaluate training quality separately from execution speed.

## Section Status

**Practiced** — debugging checks, gradient inspection, clipping, DataLoader batching, and basic profiling.
