# PyTorch 10.1 — Tensors, Shapes, Devices & GPUs

**Roadmap:** Section 10 — PyTorch & Practical Deep Learning  
**Topic status:** Practiced

## 1. What is a tensor?

A tensor is PyTorch's fundamental data structure for storing numerical data. Neural networks use tensors for input data, weights, biases, activations, gradients, and predictions.

Examples:

```python
import torch

scalar = torch.tensor(7)
vector = torch.tensor([2, 4, 6])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

- Scalar: 0 dimensions
- Vector: 1 dimension
- Matrix: 2 dimensions
- Higher-dimensional tensor: 3 or more dimensions

## 2. Shape, dimensions and number of elements

```python
x = torch.tensor([[10, 20, 30], [40, 50, 60]])

print(x.shape)   # torch.Size([2, 3])
print(x.ndim)    # 2
print(x.numel()) # 6
```

- `shape`: size of each dimension
- `ndim`: number of dimensions
- `numel()`: total number of elements

A tensor with shape `(100, 4)` represents 100 examples, each with 4 features.

## 3. Creating tensors

```python
a = torch.tensor([1, 2, 3])
b = torch.zeros(2, 3)
c = torch.ones(2, 3)
d = torch.randn(2, 3)
```

- `torch.tensor(...)`: creates a tensor from supplied data
- `torch.zeros(rows, columns)`: fills a tensor with zeros
- `torch.ones(rows, columns)`: fills a tensor with ones
- `torch.randn(rows, columns)`: samples values from a standard normal distribution

Random values can differ each time the code runs.

## 4. Indexing and slicing

PyTorch uses zero-based indexing.

```python
x = torch.tensor([[10, 20, 30], [40, 50, 60]])

print(x[0])     # tensor([10, 20, 30])
print(x[1])     # tensor([40, 50, 60])
print(x[0, 1])  # tensor(20)
print(x[1, 2])  # tensor(60)
print(x[:, 0])  # tensor([10, 40])
```

- `x[0]`: first row
- `x[1, 2]`: second row, third column
- `x[:, 0]`: first column; `:` means all rows

## 5. Reshaping tensors

`reshape()` changes the arrangement of dimensions while preserving the number of elements.

```python
x = torch.tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(2, 3)

print(y)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(y.shape)  # torch.Size([2, 3])
```

The total number of elements must remain unchanged:

`6 = 2 × 3`

Reshaping six elements into `(2, 4)` is invalid because that shape requires eight elements.

## 6. Tensor data types

```python
a = torch.tensor([1, 2, 3])
b = a.float()

print(a.dtype)  # torch.int64
print(b.dtype)  # torch.float32
```

Common types:

- `torch.int64`: 64-bit integers
- `torch.float32`: 32-bit floating-point numbers
- `torch.float64`: 64-bit floating-point numbers
- `torch.bool`: Boolean values

Floating-point tensors are commonly used for neural-network computations. Model inputs and parameters must have compatible data types.

## 7. What is CUDA?

CUDA stands for Compute Unified Device Architecture. It is NVIDIA's GPU-computing platform that allows compatible software, including PyTorch, to use supported NVIDIA GPUs.

- CPU: general-purpose processing
- GPU: highly parallel computation, useful for many deep-learning operations
- CUDA: enables supported software to perform computations on NVIDIA GPUs

CUDA is not the physical GPU itself.

## 8. Check GPU availability

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

My verified setup:

- PyTorch: `2.14.1+cu130`
- CUDA available: `True`
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

## 9. Selecting a device

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = torch.tensor([1.0, 2.0, 3.0], device=device)

print(x.device)
```

This selects CUDA when available; otherwise, it selects the CPU.

## 10. Moving a tensor between devices

```python
x = torch.tensor([1.0, 2.0, 3.0])
x = x.to(device)

print(x.device)
```

`x.to(device)` returns a tensor on the selected device. Assigning the result back to `x` ensures the variable refers to that tensor.

Tensors involved in an operation generally need to be on compatible devices.

## 11. How `nn.Linear` changes a shape

```python
import torch
import torch.nn as nn

X = torch.randn(32, 5)
layer = nn.Linear(5, 10)

output = layer(X)
print(output.shape)  # torch.Size([32, 10])
```

`nn.Linear(in_features, out_features)` transforms the last dimension from 5 features to 10 features. The batch size of 32 remains unchanged.

## 12. Final practical example

```python
import torch

x = torch.tensor([1, 2, 3, 4, 5, 6])

x = x.float()
y = x.reshape(2, 3)

print("Shape:", y.shape)
print("Dimensions:", y.ndim)
print("Elements:", y.numel())
print("Data type:", y.dtype)
print("First row:", y[0])
```

Expected output:

```text
Shape: torch.Size([2, 3])
Dimensions: 2
Elements: 6
Data type: torch.float32
First row: tensor([1., 2., 3.])
```

## Key takeaways

1. Tensors are the basic numerical data structures in PyTorch.
2. `shape`, `ndim`, and `numel()` describe tensor structure.
3. Indexing retrieves specific values, rows, or columns.
4. Reshaping changes dimensions without changing the element count.
5. Data types determine how values are represented.
6. CUDA allows PyTorch to use compatible NVIDIA GPUs.
7. Device placement and tensor shapes matter when building neural networks.
8. A linear layer changes the feature dimension while preserving the batch dimension.

**Next roadmap topic:** Section 10.2 — Datasets & DataLoaders.