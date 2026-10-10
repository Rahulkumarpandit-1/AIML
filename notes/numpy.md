# 03 — NumPy Notes for Machine Learning

## 1. What Is NumPy?

**NumPy (Numerical Python)** is a Python library used for numerical computing. It provides efficient arrays and mathematical operations on them.

NumPy is important in ML because it helps us:
- Store numerical data efficiently.
- Perform vector and matrix calculations.
- Apply mathematical operations to entire arrays.
- Calculate statistics.
- Work with the numerical representations used by ML algorithms.

### Install and Import

Install NumPy in your terminal if needed:

```bash
python -m pip install numpy
```

Import it in your Python file:

```python
import numpy as np
```

`np` is the conventional alias for NumPy.

---

## 2. NumPy Arrays

The main NumPy data structure is the `ndarray`, which can represent one-dimensional, two-dimensional, and higher-dimensional arrays.

### Creating Arrays

```python
import numpy as np

a = np.array([10, 20, 30, 40])
print(a)

b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
```

Output:

```text
[10 20 30 40]
[[1 2 3]
 [4 5 6]]
```

### Dimensions of Arrays

**1D array — vector**

```python
a = np.array([1, 2, 3])
```

Shape: `(3,)`

**2D array — matrix**

```python
b = np.array([[1, 2, 3],
              [4, 5, 6]])
```

Shape: `(2, 3)` — 2 rows and 3 columns.

**3D array**

```python
c = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
```

Shape: `(2, 2, 2)` — 2 blocks, each with 2 rows and 2 columns.

In ML, a 2D array commonly represents a dataset, where rows are samples and columns are features.

---

## 3. Important Array Attributes

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
```

Output:

```text
2
(2, 3)
6
int64
```

The exact integer dtype can vary by platform.

| Attribute | Meaning |
|---|---|
| `ndim` | Number of dimensions |
| `shape` | Size along each dimension |
| `size` | Total number of elements |
| `dtype` | Data type of elements |

### Specify the Data Type

```python
a = np.array([1, 2, 3], dtype=np.float32)
print(a)
print(a.dtype)
```

Output:

```text
[1. 2. 3.]
float32
```

Common types include `int32`, `int64`, `float32`, `float64`, and `bool`.

---

## 4. Creating Arrays Without Writing Every Element

### `np.zeros()`

Creates an array filled with zeros.

```python
a = np.zeros((2, 3))
print(a)
```

Output:

```text
[[0. 0. 0.]
 [0. 0. 0.]]
```

### `np.ones()`

```python
a = np.ones((2, 2))
print(a)
```

Output:

```text
[[1. 1.]
 [1. 1.]]
```

### `np.full()`

Fills an array with a specified value.

```python
a = np.full((2, 3), 7)
print(a)
```

Output:

```text
[[7 7 7]
 [7 7 7]]
```

### `np.arange()`

Creates values using a start, stop, and step. The stop value is excluded.

```python
a = np.arange(0, 10, 2)
print(a)
```

Output:

```text
[0 2 4 6 8]
```

### `np.linspace()`

Creates a specified number of evenly spaced values between two endpoints. By default, both endpoints are included.

```python
a = np.linspace(0, 10, 5)
print(a)
```

Output:

```text
[ 0.   2.5  5.   7.5 10. ]
```

**Difference:** `arange()` specifies a step size; `linspace()` specifies the number of values.

### Identity Matrix

```python
a = np.eye(3)
print(a)
```

Output:

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

An identity matrix has ones on its main diagonal and zeros elsewhere.

---

## 5. Indexing and Slicing

Indexing accesses specific elements. Python uses zero-based indexing.

### 1D Array

```python
a = np.array([10, 20, 30, 40, 50])

print(a[0])
print(a[-1])
print(a[1:4])
```

Output:

```text
10
50
[20 30 40]
```

`a[1:4]` includes index 1 but excludes index 4.

### 2D Array

```python
a = np.array([[10, 20, 30],
              [40, 50, 60]])

print(a[0, 1])
print(a[1, 2])
print(a[0])
print(a[:, 1])
```

Output:

```text
20
60
[10 20 30]
[20 50]
```

Explanation:

- `a[0, 1]`: first row, second column.
- `a[1, 2]`: second row, third column.
- `a[0]`: first row.
- `a[:, 1]`: every row, second column.

The colon `:` means select all elements along that axis.

### Slicing Rows and Columns

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(a[0:2, 1:3])
```

Output:

```text
[[2 3]
 [5 6]]
```

This selects the first two rows and the last two columns.

---

## 6. Modifying Array Elements

```python
a = np.array([10, 20, 30])

a[1] = 99
print(a)
```

Output:

```text
[10 99 30]
```

For a 2D array:

```python
a = np.array([[1, 2],
              [3, 4]])

a[0, 1] = 100
print(a)
```

Output:

```text
[[  1 100]
 [  3   4]]
```

### Important: Views vs Copies

Basic slicing often returns a **view**, which shares data with the original array.

```python
a = np.array([10, 20, 30, 40])
b = a[1:3]

b[0] = 99
print(a)
```

Output:

```text
[10 99 30 40]
```

Use `.copy()` when you need an independent array:

```python
a = np.array([10, 20, 30, 40])
b = a[1:3].copy()

b[0] = 99
print(a)
print(b)
```

Output:

```text
[10 20 30 40]
[99 30]
```

---

## 7. Array Operations

NumPy performs element-wise operations, meaning the operation is applied to corresponding elements.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
```

Output:

```text
[5 7 9]
[-3 -3 -3]
[ 4 10 18]
[0.25 0.4  0.5 ]
[1 4 9]
```

### Operations with Scalars

```python
a = np.array([1, 2, 3])

print(a + 10)
print(a * 2)
print(a ** 2)
```

Output:

```text
[11 12 13]
[2 4 6]
[1 4 9]
```

These operations are generally faster and more concise than writing Python loops for the same numerical work.

---

## 8. Universal Functions (ufuncs)

NumPy provides optimized functions that operate element-wise on arrays.

```python
a = np.array([1, 4, 9, 16])

print(np.sqrt(a))
print(np.exp(a))
print(np.log(a))
print(np.abs([-2, -5, 3]))
```

Common functions:

| Function | Purpose |
|---|---|
| `np.sqrt(a)` | Square root |
| `np.exp(a)` | Exponential \(e^a\) |
| `np.log(a)` | Natural logarithm |
| `np.abs(a)` | Absolute value |
| `np.round(a)` | Round values |
| `np.maximum(a, b)` | Element-wise maximum |
| `np.minimum(a, b)` | Element-wise minimum |

`np.log()` expects positive inputs for finite real-valued results.

---

## 9. Broadcasting

**Broadcasting** allows NumPy to perform operations between arrays of compatible shapes without manually copying values.

### Example 1: Array and Scalar

```python
a = np.array([10, 20, 30])
print(a + 5)
```

Output:

```text
[15 25 35]
```

The scalar `5` is applied to every element.

### Example 2: 2D Array and 1D Array

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([10, 20, 30])

print(a + b)
```

Output:

```text
[[11 22 33]
 [14 25 36]]
```

NumPy applies `b` to each row.

### Broadcasting Rules

Compare shapes from right to left. Two dimensions are compatible when:

- They are equal, or
- One of them is `1`.

Missing dimensions are treated as if their size were `1`.

For example:

- `(2, 3)` and `(3,)` are compatible.
- `(2, 3)` and `(2,)` are generally not compatible.
- `(2, 3)` and `(2, 1)` are compatible.

Broadcasting is widely used in feature scaling and numerical ML calculations.

---

## 10. Reshaping Arrays

`reshape()` changes an array's shape without changing the order of its elements.

```python
a = np.arange(1, 7)

b = a.reshape(2, 3)
print(b)
```

Output:

```text
[[1 2 3]
 [4 5 6]]
```

The new shape must contain the same total number of elements.

### Using `-1`

NumPy can infer one dimension automatically.

```python
a = np.arange(1, 7)
print(a.reshape(3, -1))
```

Output:

```text
[[1 2]
 [3 4]
 [5 6]]
```

Only one dimension can be inferred with `-1` in a single reshape operation.

### Flattening

```python
a = np.array([[1, 2],
              [3, 4]])

print(a.flatten())
```

Output:

```text
[1 2 3 4]
```

`flatten()` returns a copy. `ravel()` returns a flattened view when possible.

---

## 11. Transpose

Transpose swaps the axes of an array. For a 2D matrix, it swaps rows and columns.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.T)
```

Output:

```text
[[1 4]
 [2 5]
 [3 6]]
```

The shape changes from `(2, 3)` to `(3, 2)`.

**Why it matters:** Transpose is useful in matrix operations and in aligning dimensions for ML calculations.

---

## 12. Aggregation and Statistical Functions

Aggregation reduces multiple values to a summary.

```python
a = np.array([10, 20, 30, 40])

print(np.sum(a))
print(np.mean(a))
print(np.median(a))
print(np.min(a))
print(np.max(a))
print(np.std(a))
print(np.var(a))
```

Output:

```text
100
25.0
25.0
10
40
11.180339887498949
125.0
```

By default, `np.var()` and `np.std()` use the population convention (`ddof=0`).

### Using `axis`

Consider:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
```

Output:

```text
[5 7 9]
[ 6 15]
```

For this 2D array:

- `axis=0`: combine values down the rows, producing one result per column.
- `axis=1`: combine values across the columns, producing one result per row.

This distinction is important when calculating feature statistics across a dataset.

---

## 13. Boolean Filtering

Boolean filtering selects elements that satisfy a condition.

```python
a = np.array([10, 25, 30, 15, 40])

print(a > 20)
print(a[a > 20])
```

Output:

```text
[False  True  True False  True]
[25 30 40]
```

### Multiple Conditions

Use `&` for AND and `|` for OR. Put each comparison inside parentheses.

```python
a = np.array([10, 20, 30, 40, 50])

print(a[(a >= 20) & (a <= 40)])
print(a[(a < 20) | (a > 40)])
```

Output:

```text
[20 30 40]
[10 50]
```

Do not use Python's `and` or `or` directly between NumPy arrays.

### Replace Values Using a Condition

```python
a = np.array([10, 20, 30, 40])

a[a > 25] = 0
print(a)
```

Output:

```text
[10 20  0  0]
```

---

## 14. `np.where()`

`np.where()` can select values according to a condition.

```python
a = np.array([10, 20, 30, 40])

result = np.where(a >= 25, 1, 0)
print(result)
```

Output:

```text
[0 0 1 1]
```

Syntax:

```python
np.where(condition, value_if_true, value_if_false)
```

It is useful for simple conditional transformations.

---

## 15. Sorting and Unique Values

### Sorting

```python
a = np.array([40, 10, 30, 20])

print(np.sort(a))
```

Output:

```text
[10 20 30 40]
```

`np.sort(a)` returns a sorted copy. `a.sort()` sorts the array in place.

### Unique Values

```python
a = np.array([1, 2, 2, 3, 3, 3])

print(np.unique(a))
```

Output:

```text
[1 2 3]
```

With `return_counts=True`:

```python
values, counts = np.unique(a, return_counts=True)

print(values)
print(counts)
```

Output:

```text
[1 2 3]
[1 2 3]
```

This is useful for inspecting categories or counting occurrences.

---

## 16. Joining and Splitting Arrays

### Joining

`np.concatenate()` joins arrays along an existing axis.

```python
a = np.array([1, 2])
b = np.array([3, 4])

print(np.concatenate((a, b)))
```

Output:

```text
[1 2 3 4]
```

For 2D arrays, the dimensions other than the joining axis must match.

### Splitting

```python
a = np.array([1, 2, 3, 4, 5, 6])

print(np.split(a, 3))
```

Output:

```text
[array([1, 2]), array([3, 4]), array([5, 6])]
```

`np.split()` requires equal-sized sections when given an integer number of sections. Use `np.array_split()` when sections may have different sizes.

---

## 17. Random Numbers

Random values are useful for experiments, simulations, and creating sample data.

Use the modern random-number generator:

```python
rng = np.random.default_rng(42)

print(rng.integers(1, 10, size=5))
print(rng.random(3))
```

- `integers(1, 10, size=5)` produces five integers from 1 up to, but not including, 10.
- `random(3)` produces three random floating-point values from 0 up to, but not including, 1.

The seed `42` makes results reproducible for the same generator and sequence of operations.

### Random Normal Values

```python
rng = np.random.default_rng(42)

a = rng.normal(loc=0, scale=1, size=5)
print(a)
```

This generates five samples from a normal distribution with mean `0` and standard deviation `1`.

---

## 18. Linear Algebra for ML

NumPy provides `np.linalg` for common linear algebra operations.

### Element-wise Multiplication vs Matrix Multiplication

```python
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

print(a * b)
print(a @ b)
```

Output:

```text
[[ 5 12]
 [21 32]]

[[19 22]
 [43 50]]
```

- `a * b`: multiplies corresponding elements.
- `a @ b`: performs matrix multiplication.

Matrix multiplication requires compatible shapes: the number of columns in the first matrix must equal the number of rows in the second.

### Dot Product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b))
```

Output:

```text
32
```

Calculation: \(1(4)+2(5)+3(6)=32\).

For 1D vectors, `np.dot()` calculates the dot product. For matrix multiplication, `@` is usually clearer.

### Transpose and Shape

Before matrix multiplication, check shapes:

```python
a = np.ones((2, 3))
b = np.ones((3, 4))

print((a @ b).shape)
```

Output:

```text
(2, 4)
```

### Other Useful Operations

```python
a = np.array([[1, 2],
              [3, 4]])

print(np.linalg.det(a))
print(np.linalg.inv(a))
print(np.linalg.norm(a))
```

- `det()`: determinant of a square matrix.
- `inv()`: matrix inverse, when the matrix is invertible.
- `norm()`: vector or matrix norm.

These are useful to recognize, but advanced linear algebra can be learned when a particular algorithm requires it.

---

## 19. Copying Arrays

Assignment does not create an independent array.

```python
a = np.array([1, 2, 3])
b = a

b[0] = 100
print(a)
```

Output:

```text
[100   2   3]
```

Both variables refer to the same array.

Use `.copy()` to create an independent copy:

```python
a = np.array([1, 2, 3])
b = a.copy()

b[0] = 100

print(a)
print(b)
```

Output:

```text
[1 2 3]
[100   2   3]
```

---

## 20. Missing and Invalid Numerical Values

NumPy supports special floating-point values such as `NaN`, which means “Not a Number.”

```python
a = np.array([10.0, np.nan, 30.0])

print(np.isnan(a))
print(np.nanmean(a))
print(np.nanmax(a))
```

Output:

```text
[False  True False]
20.0
30.0
```

- `np.isnan(a)`: identifies `NaN` values.
- `np.nanmean(a)`: calculates the mean while ignoring `NaN`.
- `np.nanmax(a)`: calculates the maximum while ignoring `NaN`.

These functions do not replace proper missing-value handling in an ML preprocessing pipeline.

---

## 21. NumPy and Machine Learning

NumPy is useful for understanding the calculations behind ML models.

### Example: Linear Regression Prediction

A simple linear model predicts:

\[
\hat y=wx+b
\]

Using NumPy:

```python
import numpy as np

X = np.array([1000, 1500, 2000])

w = 2000
b = 50000

predictions = w * X + b

print(predictions)
```

Output:

```text
[2050000 3050000 4050000]
```

Here, `X` contains house areas, `w` is the weight, and `b` is the bias. The values are illustrative; they are not learned from data.

### Example: Calculate Mean Squared Error

```python
import numpy as np

actual = np.array([100, 200, 300])
predicted = np.array([110, 190, 280])

mse = np.mean((actual - predicted) ** 2)

print(mse)
```

Output:

```text
233.33333333333334
```

Calculation:

\[
MSE=\frac{(100-110)^2+(200-190)^2+(300-280)^2}{3}
\]

\[
MSE=\frac{100+100+400}{3}=200
\]

The correct output is `200.0`. The code is correct; the output line above should read:

```text
200.0
```

### Example: Feature Scaling with NumPy

Standardization transforms a feature using its mean and standard deviation:

\[
z=\frac{x-\mu}{\sigma}
\]

```python
import numpy as np

X = np.array([10.0, 20.0, 30.0])

mean = np.mean(X)
std = np.std(X)

X_scaled = (X - mean) / std

print(X_scaled)
```

Output approximately:

```text
[-1.22474487  0.          1.22474487]
```

This is a simplified demonstration. In real ML workflows, fit the scaler on training data only and reuse it for validation, test, and future data.

---

## 22. Common Mistakes

1. Confusing `shape` with `size`.
2. Forgetting that indexing starts at zero.
3. Forgetting that a slice's stop index is excluded.
4. Confusing `*` with `@`.
5. Using incompatible array shapes.
6. Using `and`/`or` instead of `&`/`|` for element-wise Boolean conditions.
7. Modifying a view when an independent copy was intended.
8. Fitting preprocessing statistics on the full dataset before splitting.
9. Assuming all NumPy operations return a new array.
10. Ignoring data types, especially when mixing integers and floating-point values.

---

## 23. Quick Revision Cheat Sheet

| Task | NumPy syntax |
|---|---|
| Create array | `np.array([1, 2, 3])` |
| Zeros | `np.zeros((2, 3))` |
| Ones | `np.ones((2, 3))` |
| Range | `np.arange(0, 10, 2)` |
| Evenly spaced values | `np.linspace(0, 1, 5)` |
| Dimensions | `a.ndim` |
| Shape | `a.shape` |
| Element count | `a.size` |
| Data type | `a.dtype` |
| Slice | `a[1:4]` |
| Select a matrix column | `a[:, 0]` |
| Reshape | `a.reshape(2, 3)` |
| Transpose | `a.T` |
| Element-wise multiplication | `a * b` |
| Matrix multiplication | `a @ b` |
| Sum / mean | `np.sum(a)` / `np.mean(a)` |
| Standard deviation | `np.std(a)` |
| Boolean filtering | `a[a > 10]` |
| Conditional selection | `np.where(condition, x, y)` |
| Sort | `np.sort(a)` |
| Unique values | `np.unique(a)` |
| Independent copy | `a.copy()` |
| Reproducible random generator | `np.random.default_rng(42)` |
| Dot product | `np.dot(a, b)` |

## 24. Practice Exercises

Try these without looking at the examples first.

1. Create a NumPy array containing numbers from 1 to 10.
2. Print its shape, dimensions, and data type.
3. Select the third element and the last three elements.
4. Create a 3 × 3 matrix and print its second row and first column.
5. Select all values greater than 15 from an array.
6. Replace all negative values with zero using `np.where()`.
7. Calculate the mean, median, minimum, maximum, and standard deviation of an array.
8. Reshape numbers 1 to 12 into a 3 × 4 matrix.
9. Add two matrices and compare `*` with `@`.
10. Calculate MSE from actual and predicted values.
11. Standardize a numerical array using its mean and standard deviation.
12. Demonstrate the difference between assigning an array to another variable and using `.copy()`.

**Completion goal:** You should be able to create arrays, understand their shapes, index and filter them, perform vectorized calculations, and use basic matrix operations. These skills are enough to support your next ML lessons.