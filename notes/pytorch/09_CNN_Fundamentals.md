# 09 — CNN Fundamentals

## 1. What is a Convolutional Neural Network?

A **Convolutional Neural Network (CNN)** is a neural network architecture designed primarily for image and other spatial data.

Images contain spatial relationships. Nearby pixels often belong to the same edge, texture, or object.

A fully connected neural network can process image pixels, but it usually requires flattening the image first. This makes it harder to preserve and learn local spatial patterns efficiently.

CNNs use convolutional filters to learn visual features directly from image data.

Examples of visual features:

- Edges and lines
- Corners
- Textures
- Shapes
- Object parts
- Complete objects

CNNs learn increasingly complex features as information passes through their layers.

## 2. Understanding Image Tensors

A colour image generally contains three channels:

- Red
- Green
- Blue

PyTorch represents an image using:

`[Channels, Height, Width]`

For a CIFAR-10 image:

`[3, 32, 32]`

This means:
- 3 colour channels
- 32 pixels in height
- 32 pixels in width

For a batch of 64 images:

`[64, 3, 32, 32]`

The first dimension represents the batch size.

**Important:** PyTorch's usual image tensor format is NCHW:

`[Batch, Channels, Height, Width]`

## 3. Convolution Layer

The convolution layer is the main building block of a CNN.

Example:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)
```

### Understanding the parameters

| Parameter | Meaning |
|---|---|
| `in_channels=3` | Input has RGB channels |
| `out_channels=16` | Creates 16 output feature maps |
| `kernel_size=3` | Uses 3 × 3 filters |
| `padding=1` | Adds one pixel of padding |
| `stride=1` | Moves the filter one position at a time |

### 3.1 What is a filter or kernel?

A filter is a small matrix of learnable weights.

For a 3 × 3 filter:

```text
w1  w2  w3
w4  w5  w6
w7  w8  w9
```

The filter moves across the input image.

At each location, it combines its weights with the corresponding local image values and produces an output value.

This process is called convolution.

Different filters learn different patterns.

For example, one filter may respond strongly to horizontal edges, while another may respond to vertical edges.

These patterns are learned during training through backpropagation.

### 3.2 What is a feature map?

A feature map is the output produced by applying a filter to the input.

If a convolution layer has 16 output channels, it produces 16 feature maps.

These maps represent different learned responses to visual patterns.

Remember the distinction:

- **Filter:** The learned weights.
- **Feature map:** The output produced by applying those weights.

### 3.3 Why do we use multiple filters?

An image contains many different visual patterns.

A single filter cannot capture all of them.

Using multiple filters allows a CNN to learn different types of features simultaneously.

For example:

```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

The layer receives three channels and produces 16 feature maps.

The next convolution layer can combine information from those 16 channels to learn more complex patterns.

## 4. Convolution Output Shape

The output dimension of a convolution is calculated using:

\[
O = \left\lfloor\frac{W-K+2P}{S}\right\rfloor+1
\]

Where:

- \(W\): Input width or height
- \(K\): Kernel size
- \(P\): Padding
- \(S\): Stride
- \(O\): Output width or height

For our first convolution:

- Input = 32
- Kernel = 3
- Padding = 1
- Stride = 1

\[
O = \frac{32-3+2(1)}{1}+1=32
\]

Therefore, the spatial dimensions remain 32 × 32.

The output channels change from 3 to 16.

Input:

`[3, 32, 32]`

Output:

`[16, 32, 32]`

For a batch of 64:

`[64, 16, 32, 32]`

### Effect of padding

- `padding=0`: Spatial dimensions generally decrease.
- `padding=1` with a 3 × 3 kernel and stride 1: Spatial dimensions remain unchanged.
- Larger padding can increase spatial dimensions.

Padding helps control the output dimensions and allows filters to process regions near image boundaries.

## 5. ReLU Activation Function

ReLU stands for Rectified Linear Unit.

Its mathematical definition is:

\[
ReLU(x)=\max(0,x)
\]

Examples:

| Input | Output |
|---:|---:|
| -5 | 0 |
| -2 | 0 |
| 0 | 0 |
| 3 | 3 |
| 8 | 8 |

PyTorch:

```python
nn.ReLU()
```

### Why is ReLU necessary?

Convolution layers perform linear operations.

If we stack only linear transformations, the overall model remains a linear transformation.

ReLU introduces non-linearity, allowing the network to learn more complex relationships.

ReLU does not change the tensor dimensions.

Example:

`[16, 32, 32] → [16, 32, 32]`

## 6. Max Pooling

Pooling reduces the spatial dimensions of feature maps.

Example:

```python
nn.MaxPool2d(kernel_size=2)
```

A 2 × 2 pooling window selects the maximum value from each region.

Example:

```text
1  3
2  4
```

Maximum value:

```text
4
```

With the default stride of 2, the pooling window moves two positions at a time.

Therefore, height and width are approximately halved when the dimensions are even.

Example:

`[16, 32, 32] → [16, 16, 16]`

### Why use Max Pooling?

- Reduces spatial dimensions.
- Reduces computation in later layers.
- Preserves strong local responses.
- Makes features somewhat less sensitive to small positional changes.

Pooling does not change the number of channels.

## 7. Flatten Layer

After convolution and pooling, the CNN has multidimensional feature maps.

A fully connected layer expects a feature vector for each image.

Flatten converts the feature maps into a one-dimensional vector.

Example:

Before flattening:

`[32, 8, 8]`

After flattening:

`[2048]`

Calculation:

\[
32 \times 8 \times 8 = 2048
\]

PyTorch:

```python
nn.Flatten()
```

Flatten does not learn weights. It changes the arrangement of the data.

For a batch of 64:

`[64, 32, 8, 8] → [64, 2048]`

## 8. Linear Layer

The Linear layer combines extracted features and produces output scores.

Example:

```python
nn.Linear(32 * 8 * 8, 10)
```

The parameters mean:

- 2048 input features.
- 10 output values.

For CIFAR-10, the ten outputs correspond to ten classes.

The outputs are called **logits**.

For a batch of 64:

`[64, 2048] → [64, 10]`

The Linear layer learns weights that combine the extracted features to produce class scores.

## 9. Complete CNN Architecture

Our initial architecture:

```python
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10)
).to(device)
```

### Layer-by-layer shape calculation

For one image:

| Layer | Output shape |
|---|---|
| Input | `[3, 32, 32]` |
| Conv2d(3, 16, 3, padding=1) | `[16, 32, 32]` |
| ReLU | `[16, 32, 32]` |
| MaxPool2d(2) | `[16, 16, 16]` |
| Conv2d(16, 32, 3, padding=1) | `[32, 16, 16]` |
| ReLU | `[32, 16, 16]` |
| MaxPool2d(2) | `[32, 8, 8]` |
| Flatten | `[2048]` |
| Linear(2048, 10) | `[10]` |

For a batch of 64, add 64 at the beginning of each shape.

### Understanding the architecture

```text
Input Image
    |
    v
Convolution
    |
    v
ReLU
    |
    v
Max Pooling
    |
    v
Convolution
    |
    v
ReLU
    |
    v
Max Pooling
    |
    v
Flatten
    |
    v
Linear Layer
    |
    v
10 Class Scores
```

A useful way to remember the operations:

- Convolution: Find features.
- ReLU: Introduce non-linearity.
- Pooling: Reduce spatial dimensions.
- Flatten: Convert feature maps into vectors.
- Linear: Produce class scores.

## 10. Important CNN Concepts

| Concept | Meaning |
|---|---|
| Kernel/filter | Learnable weights that process local image regions |
| Feature map | Output produced by a convolution filter |
| Channels | Different feature representations |
| Padding | Adds pixels around the input boundary |
| Stride | Distance the filter moves each step |
| ReLU | Non-linear activation function |
| Max Pooling | Reduces spatial dimensions |
| Flatten | Converts feature maps into vectors |
| Linear | Produces output scores |

## 11. Common Mistakes

### Mistake 1: Incorrect number of input channels

If an image has RGB channels, the first convolution generally needs three input channels.

```python
nn.Conv2d(3, 16, kernel_size=3)
```

### Mistake 2: Incorrect Linear input size

The Linear layer's input dimension must match the flattened output.

If the feature maps have shape `[32, 8, 8]`, the input size is 2048.

### Mistake 3: Confusing channels with image dimensions

In:

`[64, 32, 8, 8]`

- 64 = batch size
- 32 = channels
- 8 = height
- 8 = width

### Mistake 4: Thinking filters are manually designed

In ordinary CNN training, filter weights are learned from data using gradients. We choose the filter dimensions and number of filters, not the final learned values.

## 12. Practical Understanding Checklist

- [ ] Explain why CNNs are useful for image classification.
- [ ] Explain how a convolution filter works.
- [ ] Explain the difference between a filter and a feature map.
- [ ] Calculate convolution output dimensions.
- [ ] Explain the role of padding and stride.
- [ ] Explain why ReLU is used.
- [ ] Calculate the output shape after MaxPool2d.
- [ ] Explain Flatten and Linear.
- [ ] Calculate the correct Linear input size.
- [ ] Build the basic CNN architecture independently.

**Final takeaway:** A CNN progressively converts raw image pixels into learned feature representations, then uses those representations to produce class scores.