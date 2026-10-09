# 02 — Mathematics for Machine Learning

## 1. Basic Algebra

Algebra helps us understand the equations used by ML models.

### Variables and Equations

A variable represents a value.

Example:

\[
y=2x+3
\]

If \(x=4\):

\[
y=2(4)+3=11
\]

### Functions

A function maps an input to an output.

\[
f(x)=2x+3
\]

In ML, a model behaves like a function that takes features as input and produces a prediction.

### Summation

The symbol \(\sum\) means adding multiple values.

\[
\sum_{i=1}^{3}x_i=x_1+x_2+x_3
\]

Example: If the values are 2, 4, and 6, their sum is 12.

Summation is used in loss functions and statistical calculations.

---

## 2. Statistics

Statistics helps us understand data before training models.

### Mean

The average of all values.

\[
\text{Mean}=\frac{\text{Sum of values}}{\text{Number of values}}
\]

Example: Values 2, 4, 6.

Mean \(=(2+4+6)/3=4\).

### Median

The middle value after sorting the data.

Example: 2, 5, 8 → Median = 5.

The median is often more robust to extreme values than the mean.

### Variance

Variance measures how spread out values are around their mean.

For a population:

\[
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
\]

Where:
- \(N\): number of values
- \(x_i\): individual value
- \(\mu\): mean

### Standard Deviation

Standard deviation is the square root of variance.

\[
\sigma=\sqrt{\sigma^2}
\]

It describes spread in the original unit of the data.

**Why it matters:** These concepts help with understanding datasets, detecting unusual values, and scaling features.

---

## 3. Probability

Probability measures how likely an event is.

\[
0\leq P(A)\leq 1
\]

- 0 means impossible.
- 1 means certain.
- 0.5 means a 50% probability.

Example: For a fair coin, the probability of heads is \(1/2=0.5\).

### Conditional Probability

The probability of event \(A\) given that event \(B\) has occurred.

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
\]

This applies when \(P(B)>0\).

**Why it matters:** Probability is important for classification, uncertainty, and interpreting probabilistic model outputs.

---

## 4. Vectors and Matrices

ML datasets are commonly represented using vectors and matrices.

### Vector

A vector is an ordered collection of numbers.

\[
x=\begin{bmatrix}2\\3\\4\end{bmatrix}
\]

A row of features for one sample can be represented as a vector.

### Matrix

A matrix is a rectangular arrangement of numbers.

\[
X=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
\]

A dataset is commonly represented as a matrix:
- Rows represent samples.
- Columns represent features.

### Dot Product

For two vectors of equal length:

\[
a\cdot b=\sum_{i=1}^{n}a_ib_i
\]

Example:

\[
[2,3]\cdot[4,5]=(2\times4)+(3\times5)=23
\]

Dot products are used in Linear Regression and neural networks.

**Priority:** Understand vectors, matrices, shapes, and dot products. Advanced matrix algebra can wait until you need it.

---

## 5. Derivatives and Gradients

These concepts are essential for understanding gradient descent.

### Derivative

A derivative measures how quickly a function changes with respect to its input.

For example:

\[
f(x)=x^2
\]

Its derivative is:

\[
f'(x)=2x
\]

At \(x=3\):

\[
f'(3)=6
\]

The derivative tells us the local rate of change at that point.

### Basic Rules

Power rule:

\[
\frac{d}{dx}x^n=nx^{n-1}
\]

Constant rule:

\[
\frac{d}{dx}c=0
\]

Constant multiple rule:

\[
\frac{d}{dx}[cf(x)]=cf'(x)
\]

Example:

\[
f(x)=3x^2+2x+5
\]

Therefore:

\[
f'(x)=6x+2
\]

### Gradient

A gradient is the collection of partial derivatives of a function with respect to its parameters.

For a function with parameters \(w\) and \(b\):

\[
L(w,b)
\]

Its gradient is:

\[
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w}\\
\frac{\partial L}{\partial b}
\end{bmatrix}
\]

It tells us how the loss changes as each parameter changes.

---

## 6. Gradient Descent

Gradient descent updates model parameters to reduce the loss.

\[
w_{\text{new}}=w_{\text{old}}-\alpha\frac{dL}{dw}
\]

Where:
- \(w\): model parameter
- \(L\): loss function
- \(\alpha\): learning rate
- \(\frac{dL}{dw}\): derivative of the loss with respect to \(w\)

### Numerical Example

Suppose:

\[
L(w)=(w-3)^2
\]

The derivative is:

\[
\frac{dL}{dw}=2(w-3)
\]

Start with \(w=0\) and learning rate \(\alpha=0.1\).

Gradient:

\[
2(0-3)=-6
\]

Update:

\[
w_{\text{new}}=0-0.1(-6)=0.6
\]

The parameter moves toward 3, where the loss reaches zero.

**Remember:** The learning rate controls the update size. If it is too large, updates can overshoot the minimum.

---

## 7. Loss Functions

A loss function measures the error between actual and predicted values.

### Mean Squared Error (MSE)

Commonly used for regression:

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Where:
- \(y_i\): actual value
- \(\hat y_i\): predicted value
- \(n\): number of samples

Example: Actual values are 3 and 5; predictions are 2 and 7.

\[
MSE=\frac{(3-2)^2+(5-7)^2}{2}=2.5
\]

A lower MSE indicates smaller squared prediction errors on the evaluated data.

Classification uses other losses, such as cross-entropy, which you can learn when studying classification.

---

## 8. Essential Math for ML: Priority Order

Learn in this order:

1. Algebra and functions
2. Mean, median, variance, and standard deviation
3. Basic probability
4. Vectors, matrices, and dot products
5. Derivatives and partial derivatives
6. Gradient descent and loss functions

You do **not** need to master advanced calculus or every topic in linear algebra before starting practical ML. Learn the mathematics needed for each algorithm as you implement it.

## Quick Revision

- **Mean:** Average value.
- **Variance:** Measures spread around the mean.
- **Standard deviation:** Square root of variance.
- **Probability:** Measures how likely an event is.
- **Vector:** Ordered collection of numbers.
- **Matrix:** Rectangular arrangement of numbers.
- **Dot product:** Sum of pairwise products of vector elements.
- **Derivative:** Local rate of change.
- **Gradient:** Partial derivatives with respect to multiple parameters.
- **Loss:** Measures prediction error.
- **Gradient descent:** Updates parameters to reduce loss.
- **Learning rate:** Controls the size of each update.