# CNN Fundamentals with PyTorch
## CIFAR-10 Image Classification — Concepts, Implementation, Experiments and Evaluation

---

## 1. What is a Convolutional Neural Network (CNN)?

A **Convolutional Neural Network (CNN)** is a type of neural network designed primarily for data with a spatial structure, especially images.

A normal fully connected neural network treats input features as a flat list of numbers. A CNN, on the other hand, preserves the spatial arrangement of image pixels and learns patterns from nearby regions.

For example, an image contains:

- Edges and lines
- Textures
- Shapes
- Object parts
- Complete objects

A CNN learns these patterns progressively through its layers.

### Why not use only a normal neural network?

Consider a colour image of size 32 × 32 pixels.

It contains:

\[
32 \times 32 \times 3 = 3072
\]

individual pixel values.

If we immediately flatten this image and feed it into a fully connected network, we lose the explicit two-dimensional arrangement of pixels.

A CNN processes the image while retaining its spatial structure.

**Core idea:** CNNs learn visual features automatically instead of requiring us to manually define image features.

---

# 2. Understanding the CIFAR-10 Dataset

We used the CIFAR-10 dataset to build our image classification model.

It contains 60,000 colour images divided into:

| Dataset | Number of images |
|---|---:|
| Training set | 50,000 |
| Test set | 10,000 |
| Total | 60,000 |

Each image belongs to one of ten classes.

### The ten classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Each image has:

- Height: 32 pixels
- Width: 32 pixels
- Channels: 3 (Red, Green, Blue)

### Image tensor shape

PyTorch generally represents an image using:

`[Channels, Height, Width]`

Therefore:

`[3, 32, 32]`

A batch of 64 images has this shape:

`[64, 3, 32, 32]`

The first dimension represents the number of images in the batch.

### Why use batches?

Instead of processing all 50,000 training images at once, we divide them into smaller groups.

With a batch size of 64:

\[
\text{Batches per epoch}
=
\left\lceil \frac{50000}{64} \right\rceil
= 782
\]

An epoch means one complete pass through the training dataset.

---

# 3. Understanding the CNN Architecture

Our initial CNN architecture:

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

Let's understand each layer and how information moves through the model.

## 3.1 Conv2d — Convolution Layer

```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

A convolution layer uses learnable filters, also called kernels, to detect patterns in an image.

The parameters mean:

- `3`: Number of input channels (RGB).
- `16`: Number of output channels or filters.
- `kernel_size=3`: Each filter has a 3 × 3 spatial size.
- `padding=1`: Adds one pixel of padding around the image.

### What does a filter do?

Imagine a 3 × 3 filter moving across an image.

At each position, it combines the values in the local image region with its learned weights.

This operation produces a value in an output feature map.

Different filters can learn to detect different patterns, such as:

- Horizontal edges
- Vertical edges
- Corners
- Textures
- More complex visual patterns

We do not manually tell the CNN what each filter should detect. Training adjusts the filter weights using gradients.

### What is a feature map?

A feature map is the output produced by a filter.

If a convolution layer has 16 filters, it produces 16 output feature maps.

These maps represent different learned visual responses.

**Important distinction:** A filter contains learned weights. A feature map is the result of applying that filter to an input.

## 3.2 Understanding output dimensions

The output size of a convolution is:

\[
O = \frac{W-K+2P}{S}+1
\]

Where:

- \(W\): Input width or height
- \(K\): Kernel size
- \(P\): Padding
- \(S\): Stride
- \(O\): Output width or height

For our first convolution:

- Input size = 32
- Kernel size = 3
- Padding = 1
- Stride = 1

\[
O = \frac{32-3+2(1)}{1}+1=32
\]

Therefore, the height and width remain 32.

The number of output channels changes from 3 to 16.

Output shape:

`[16, 32, 32]`

## 3.3 ReLU — Activation Function

```python
nn.ReLU()
```

ReLU stands for Rectified Linear Unit.

Its operation is:

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

ReLU replaces negative values with zero while retaining positive values.

### Why do we need ReLU?

Without activation functions, stacking multiple linear operations would still produce an overall linear transformation.

ReLU introduces non-linearity, allowing the network to learn more complex patterns.

**Remember:** Convolution extracts learned features; ReLU introduces non-linearity.

ReLU does not change the tensor's dimensions.

## 3.4 MaxPool2d — Pooling Layer

```python
nn.MaxPool2d(2)
```

Max pooling reduces the spatial dimensions of a feature map.

With a 2 × 2 pooling window and the default stride of 2, it takes the maximum value from each 2 × 2 region.

For example:

```text
1  3
2  4
```

The maximum value is:

```text
4
```

The operation is repeated across the feature map.

### Why use pooling?

- Reduces image feature dimensions.
- Reduces computation in later layers.
- Keeps strong local responses.
- Makes the representation less sensitive to small spatial changes.

For our model:

`[16, 32, 32]` becomes `[16, 16, 16]`.

Pooling halves the height and width while preserving the number of channels.

---

# 4. Complete Tensor Shape Calculation

Understanding tensor shapes is essential when designing CNNs.

Our architecture processes one image as follows:

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

For a batch of 64 images, the shapes become:

| Layer | Output shape |
|---|---|
| Input | `[64, 3, 32, 32]` |
| First convolution | `[64, 16, 32, 32]` |
| First pooling | `[64, 16, 16, 16]` |
| Second convolution | `[64, 32, 16, 16]` |
| Second pooling | `[64, 32, 8, 8]` |
| Flatten | `[64, 2048]` |
| Linear | `[64, 10]` |

### Why is the flattened size 2048?

After the second pooling layer:

`[32, 8, 8]`

Flatten combines the channels, height and width into one dimension:

\[
32 \times 8 \times 8 = 2048
\]

Therefore:

```python
nn.Flatten(),
nn.Linear(32 * 8 * 8, 10)
```

The linear layer receives 2048 features and produces 10 outputs.

---

# 5. Flatten and Linear Layer

## 5.1 Flatten

```python
nn.Flatten()
```

Flatten converts multidimensional feature maps into a one-dimensional feature vector for each image.

It does not learn anything and does not change the actual values. It changes their arrangement.

Before:

`[32, 8, 8]`

After:

`[2048]`

## 5.2 Linear Layer

```python
nn.Linear(32 * 8 * 8, 10)
```

The linear layer combines the extracted features to produce ten output values.

Each output corresponds to one CIFAR-10 class.

For a batch of 64 images:

`[64, 2048] → [64, 10]`

The ten output values are called **logits**.

### What are logits?

Logits are raw model scores before softmax.

For one image, the model might produce:

```text
Airplane:    2.1
Automobile:  0.5
Bird:       -1.2
Cat:         0.8
Deer:        0.2
Dog:         1.4
Frog:       -0.5
Horse:       0.3
Ship:        0.9
Truck:       0.1
```

The largest logit is the model's predicted class.

In this example, the prediction is Airplane.

We can obtain the predicted class using:

```python
predictions = logits.argmax(dim=1)
```

### Logits versus probabilities

Logits are not probabilities.

Softmax converts logits into values between 0 and 1 that sum to 1:

```python
probabilities = torch.softmax(logits, dim=1)
```

We do not need to apply softmax before passing logits to `nn.CrossEntropyLoss`.

---

# 6. How CNN Training Actually Works

A neural network learns by repeatedly:

1. Receiving input images.
2. Producing predictions.
3. Comparing predictions with correct labels.
4. Calculating loss.
5. Computing gradients.
6. Updating weights.

This complete process is called training.

## 6.1 Forward pass

```python
logits = model(images)
```

Images pass through all model layers.

The model produces ten logits for every image.

This is called the forward pass.

## 6.2 Loss function

```python
loss_fn = nn.CrossEntropyLoss()
```

Cross-entropy loss measures how far the model's predictions are from the correct class labels.

For example:

- Actual label: Cat
- Model's highest score: Dog

The model receives a higher loss because it predicted the wrong class.

If the model assigns a high probability to the correct class, the loss is generally lower.

For classification, we use:

```python
loss = loss_fn(logits, labels)
```

`logits` contains raw output scores.

`labels` contains the correct class indices.

**Important:** `CrossEntropyLoss` expects raw logits, not probabilities produced by softmax.

## 6.3 Backpropagation

```python
loss.backward()
```

Backpropagation calculates gradients of the loss with respect to the model's trainable parameters.

In simple terms, it calculates how each weight contributed to the prediction error.

These gradients tell the optimizer which direction the weights should move to reduce the loss.

Backpropagation uses the chain rule of differentiation.

## 6.4 Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

An optimizer updates the model's trainable weights using the calculated gradients.

Adam maintains moving estimates of gradients and squared gradients to adapt the size of parameter updates.

The learning rate controls the general scale of those updates.

A learning rate that is too high can make training unstable. One that is too low can make training very slow.

## 6.5 Complete training step

```python
optimizer.zero_grad()

logits = model(images)

loss = loss_fn(logits, labels)

loss.backward()

optimizer.step()
```

Understand the exact order:

| Code | Purpose |
|---|---|
| `optimizer.zero_grad()` | Clears gradients from the previous batch |
| `model(images)` | Performs forward pass |
| `loss_fn(...)` | Calculates prediction error |
| `loss.backward()` | Calculates gradients |
| `optimizer.step()` | Updates model parameters |

### Why do we need zero_grad?

PyTorch accumulates gradients by default.

If we do not clear old gradients, the next batch's gradients are added to the previous ones.

We usually want each batch's gradients to be calculated independently, so we clear them before the next forward pass.

---

# 7. Understanding Training Accuracy and Loss

Two important measurements during training are loss and accuracy.

### Training loss

Measures how wrong the model's predictions are according to the selected loss function.

Lower loss generally indicates that the model is fitting the training data better.

### Training accuracy

Measures the percentage of training images predicted correctly.

\[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\times 100
\]

Example:

If 640 out of 1,000 images are classified correctly:

\[
Accuracy = \frac{640}{1000}\times100=64\%
\]

### Why monitor both?

A model can have:

- High accuracy but poorly calibrated confidence.
- Decreasing loss without a large accuracy improvement.
- High training accuracy but poor test accuracy.

Therefore, loss and accuracy provide different information.

---

# 8. Training Mode and Evaluation Mode

PyTorch has two important model modes.

## 8.1 Training mode

```python
model.train()
```

Used while training the model.

It enables training-specific behaviour in layers such as:

- Dropout
- Batch Normalization

## 8.2 Evaluation mode

```python
model.eval()
```

Used while evaluating or making predictions.

It changes the behaviour of certain layers:

- Dropout is disabled.
- BatchNorm uses its stored running statistics.

### Important: eval() versus no_grad()

These are not the same thing.

```python
model.eval()

with torch.no_grad():
    logits = model(images)
```

`model.eval()` changes the behaviour of certain layers.

`torch.no_grad()` prevents PyTorch from tracking gradients.

For ordinary evaluation and inference, we generally use both.

---

# 9. Saving and Loading a Model

After training, we save the learned parameters.

```python
torch.save(model.state_dict(), "cnn_model.pth")
```

A `state_dict` contains the model's learned parameters and relevant buffers.

It does not automatically contain the Python architecture definition.

### Loading the model

First recreate the same architecture:

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

Then load the weights:

```python
state = torch.load(
    "cnn_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state)
model.eval()
```

### Why must the architecture match?

The saved weights have specific shapes and correspond to specific model layers.

If the architecture changes, the parameter shapes or layer keys may no longer match.

For example, changing the first convolution from 16 output channels to 32 changes the expected parameter dimensions.

The architecture used for prediction must therefore match the architecture used for training.

**Remember:** Save the learned parameters, recreate the architecture, load the parameters, then switch to evaluation mode.

---

# 10. Data Normalization

Before training, we transformed image pixel values.

```python
transforms.ToTensor()
```

`ToTensor()` converts image data into PyTorch tensors and scales typical 8-bit pixel values from `[0, 255]` to `[0, 1]`.

We then used:

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

For each channel, normalization performs:

\[
x_{\text{normalized}} =
\frac{x-\text{mean}}{\text{standard deviation}}
\]

With mean 0.5 and standard deviation 0.5:

\[
x_{\text{normalized}}=\frac{x-0.5}{0.5}
\]

Values in `[0, 1]` are therefore mapped approximately to `[-1, 1]`.

### Why normalize images?

- Provides a more consistent input scale.
- Often makes optimization easier.
- Helps neural networks train more reliably.

The same normalization must be applied during testing and prediction.

---

# 11. Data Augmentation

Data augmentation creates varied versions of training images using transformations.

We used:

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

Let's understand the transformations.

## 11.1 RandomCrop

```python
transforms.RandomCrop(32, padding=4)
```

Adds padding and then randomly crops the image back to 32 × 32.

This produces slightly different image positions.

## 11.2 RandomHorizontalFlip

```python
transforms.RandomHorizontalFlip()
```

Randomly flips an image horizontally.

For example, an automobile may appear facing the opposite direction.

This helps the model learn that small changes in orientation should not necessarily change the object category.

## 11.3 Why augment only training data?

Training augmentation helps the model learn from varied examples.

The test set should represent the data on which we want to measure generalization.

Therefore:

- Training set: random transformations + tensor conversion + normalization.
- Test set: tensor conversion + normalization only.

We should not randomly augment the test images during ordinary evaluation.

### Why did augmentation help?

Our baseline model could learn the training images but did not generalize as well to unseen test images.

Augmentation exposed the model to more variation during training.

In our experiment, test accuracy increased from 68.16% to 71.17%.

---

# 12. Dropout

Dropout is a regularization technique used to reduce overfitting.

We experimented with:

```python
nn.Flatten(),
nn.Dropout(p=0.1),
nn.Linear(32 * 8 * 8, 10)
```

### What does Dropout do?

During training, Dropout randomly sets a fraction of activations to zero.

With:

```python
nn.Dropout(p=0.1)
```

approximately 10% of the selected activations are randomly zeroed during each training forward pass.

This does not permanently delete neurons.

The dropped activations can be different on the next batch.

### Why use Dropout?

It discourages the model from relying too heavily on a small set of activations.

This can reduce overfitting and encourage more robust representations.

### What happens during evaluation?

When we call:

```python
model.eval()
```

Dropout is disabled.

The model uses the full set of activations.

### Our experiment

The Dropout p=0.1 run achieved:

**Test accuracy: 69.02%**

This was below our augmentation-only and augmentation-plus-BatchNorm results.

This does not prove that Dropout is always harmful. Its effect depends on the architecture, dropout probability, training duration and other regularization methods.

---

# 13. Batch Normalization

We added Batch Normalization after convolution layers.

```python
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10)
).to(device)
```

### What does BatchNorm do?

Batch Normalization normalizes activations using statistics calculated from a mini-batch during training.

It also learns two parameters:

- Gamma: Scale
- Beta: Shift

These allow the network to learn an appropriate output distribution after normalization.

For convolutional feature maps, we use:

```python
nn.BatchNorm2d(number_of_channels)
```

Therefore:

- After `Conv2d(3, 16, ...)`, use `BatchNorm2d(16)`.
- After `Conv2d(16, 32, ...)`, use `BatchNorm2d(32)`.

### Why place BatchNorm after convolution?

The convolution produces feature maps. BatchNorm normalizes their activations before the next non-linear activation.

Our sequence is:

```text
Convolution
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Max Pooling
```

### What happens during training?

BatchNorm uses the current batch's mean and variance and updates running statistics.

### What happens during evaluation?

BatchNorm uses its stored running statistics instead of calculating statistics from the current evaluation batch.

This is why calling `model.eval()` matters.

### Our experiment

Augmentation + BatchNorm achieved:

**Test accuracy: 72.18%**

This was our best result among the tested configurations.

---

# 14. Learning Rate Scheduling — StepLR

Initially, we used a fixed learning rate:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

A fixed learning rate can work, but a model may benefit from smaller updates as training progresses.

A learning rate scheduler changes the learning rate according to a defined strategy.

We used StepLR:

```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)
```

### Understanding its parameters

`step_size=5`

Reduces the learning rate after every five scheduler steps.

`gamma=0.5`

Multiplies the learning rate by 0.5 at each reduction.

Starting learning rate:

`0.001`

The schedule is:

| Training epochs | Learning rate used |
|---|---:|
| 1–5 | 0.001 |
| 6–10 | 0.0005 |
| 11–15 | 0.00025 |
| After epoch 15 | 0.000125 |

### Where should scheduler.step() go?

It should be called once per epoch, after all training batches have finished.

Example:

```python
for epoch in range(num_epochs):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

    scheduler.step()
```

Do not put `scheduler.step()` inside the batch loop for this StepLR setup.

### Important detail about printed learning rates

If you print the learning rate after `scheduler.step()`, the value shown is the learning rate for the next epoch.

For example, epoch 5 trained with 0.001. After the scheduler steps at the end of epoch 5, the learning rate becomes 0.0005.

This distinction is important when interpreting training logs.

### Our StepLR experiment

We trained for 15 epochs.

The final results were:

- Training accuracy: 67.19%
- Test accuracy: 71.06%
- Training loss: 0.9508

StepLR did not improve test accuracy over our BatchNorm experiment.

A scheduler is not automatically beneficial. It must be evaluated as part of the complete training configuration.

---

# 15. Evaluation: How Do We Know Whether the Model Is Good?

Training accuracy alone is not enough.

We need to evaluate the model on data it did not use for gradient updates.

For classification, important evaluation measurements include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 15.1 Accuracy

Accuracy measures the fraction of all predictions that are correct.

\[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\]

It provides an overall summary but does not explain which classes the model struggles with.

## 15.2 Precision

Precision answers:

**When the model predicts a particular class, how often is it correct?**

\[
Precision =
\frac{TP}{TP+FP}
\]

Where:

- TP = True Positives
- FP = False Positives

Example:

If the model predicts 100 images as cats and 70 really are cats:

\[
Precision = \frac{70}{100}=0.70
\]

Precision is 70%.

## 15.3 Recall

Recall answers:

**Out of all actual examples of a class, how many did the model correctly identify?**

\[
Recall =
\frac{TP}{TP+FN}
\]

Where FN means False Negatives.

Example:

If the test set contains 100 cats and the model correctly identifies 60:

\[
Recall = \frac{60}{100}=0.60
\]

Recall is 60%.

## 15.4 F1-score

F1-score combines precision and recall using their harmonic mean.

\[
F1 =
2 \times
\frac{Precision \times Recall}
{Precision+Recall}
\]

It is useful when we want a single measure that balances precision and recall.

## 15.5 Confusion matrix

A confusion matrix shows how predictions compare with actual labels.

For a ten-class classification problem, it is a 10 × 10 matrix.

- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal entries represent correct predictions.
- Off-diagonal entries represent misclassifications.

It helps identify which classes the model confuses.

For example, the model may frequently confuse cats and dogs because they share some visual features.

---

# 16. Test Evaluation Code

A typical evaluation loop:

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")
```

### Understand the important lines

`model.eval()`

Switches the model to evaluation mode.

`torch.no_grad()`

Disables gradient tracking.

`logits = model(images)`

Obtains predictions from the model.

`argmax(dim=1)`

Selects the class with the highest logit for each image.

`(predictions == labels)`

Compares predicted labels with actual labels.

`sum().item()`

Counts the correct predictions and converts the tensor result into a Python number.

---

# 17. Understanding the Experiments We Performed

We did not stop after building the first CNN. We changed the training configuration and compared the results.

The experiments helped us understand how different techniques affect generalization.

## 17.1 Experiment comparison

| Experiment | Test accuracy |
|---|---:|
| Original CNN | 68.16% |
| Data augmentation | 71.17% |
| Augmentation + BatchNorm | **72.18%** |
| Augmentation + BatchNorm + StepLR | 71.06% |
| Augmentation + Dropout (p=0.1) | 69.02% |

<ins>Best result in these experiments: Augmentation + BatchNorm, at 72.18%.</ins>

### What did we learn?

**Original CNN — 68.16%**

Our baseline. It established a reference point for future improvements.

**Data augmentation — 71.17%**

Improved test accuracy by exposing the model to more varied training images.

Improvement over baseline:

`71.17 - 68.16 = 3.01 percentage points`

**Augmentation + BatchNorm — 72.18%**

Produced the best result in our experiments.

Improvement over baseline:

`72.18 - 68.16 = 4.02 percentage points`

**Augmentation + BatchNorm + StepLR — 71.06%**

The learning rate schedule changed training behaviour, but this configuration did not beat the BatchNorm result.

**Augmentation + Dropout (p=0.1) — 69.02%**

Dropout did not improve the result in this particular setup.

### Important experimental limitation

These are results from our particular runs, not universal conclusions about the techniques.

For a rigorous comparison, use the same train/validation/test split, training duration, random seed and evaluation procedure.

Ideally, use a validation set to select hyperparameters and reserve the test set for the final evaluation.

Repeatedly choosing configurations based on test accuracy makes the test set less independent as a final measure.

---

# 18. Why Can Training Accuracy Be Lower Than Test Accuracy?

In some of our experiments, training accuracy was lower than test accuracy.

For example, the StepLR run achieved:

- Training accuracy: 67.19%
- Test accuracy: 71.06%

This is possible.

During training:

- Random data augmentation makes images harder.
- Dropout, when present, randomly removes activations.
- BatchNorm uses training batch statistics.
- The model is actively being updated.

During evaluation:

- Random augmentation is disabled.
- Dropout is disabled.
- BatchNorm uses running statistics.
- The model's weights are fixed.

Therefore, training accuracy and test accuracy are not always directly comparable.

A higher test accuracy does not automatically mean something is wrong.

However, a large or unexpected gap should be investigated.

---

# 19. Understanding the Complete CNN Workflow

The complete workflow looks like this:

```text
CIFAR-10 Dataset
       |
       v
Load Training and Test Data
       |
       v
Apply Transforms
       |
       v
Create DataLoaders
       |
       v
Build CNN Architecture
       |
       v
Initialize Loss and Optimizer
       |
       v
       Training
       |
       v
     Forward Pass
       |
       v
    Calculate Loss
       |
       v
    Backpropagation
       |
       v
  Update Model Weights
       |
       v
 Repeat for all batches and epochs
       |
       v
 Apply Scheduler Step
       |
       v
 Evaluate on Test Data
       |
       v
 Calculate Metrics
       |
       v
 Save Model Parameters
       |
       v
 Load Model for Prediction
       |
       v
 Predict New Images
```

Understanding this workflow is more important than memorizing individual PyTorch commands.

---

# 20. Important PyTorch Concepts to Remember

| Concept | Meaning |
|---|---|
| Tensor | Multidimensional data structure |
| Dataset | Provides individual samples and labels |
| DataLoader | Loads data in batches |
| Batch size | Number of samples processed together |
| Epoch | One complete pass through training data |
| Conv2d | Learns spatial features using filters |
| Feature map | Output produced by a convolution filter |
| ReLU | Adds non-linearity |
| MaxPool2d | Reduces spatial dimensions |
| Flatten | Converts feature maps into vectors |
| Linear | Produces class scores |
| Logits | Raw output scores |
| CrossEntropyLoss | Measures classification error |
| Backpropagation | Computes gradients |
| Optimizer | Updates trainable parameters |
| Learning rate | Controls update scale |
| Scheduler | Adjusts learning rate |
| Dropout | Randomly disables activations during training |
| BatchNorm | Normalizes activations using batch/running statistics |
| `model.train()` | Enables training behaviour |
| `model.eval()` | Enables evaluation behaviour |
| `torch.no_grad()` | Disables gradient tracking |
| `state_dict()` | Stores model parameters and buffers |

---

# 21. Common Mistakes and Their Solutions

### Mistake 1: Incorrect Linear input size

If the flattened output is 2048 but the linear layer expects 1024, PyTorch raises a matrix dimension error.

**Solution:** Calculate the tensor dimensions after every convolution and pooling layer.

### Mistake 2: Applying softmax before CrossEntropyLoss

Incorrect:

```python
probabilities = torch.softmax(logits, dim=1)
loss = loss_fn(probabilities, labels)
```

Correct:

```python
loss = loss_fn(logits, labels)
```

### Mistake 3: Forgetting optimizer.zero_grad()

Gradients accumulate across batches.

**Solution:** Clear gradients before each training batch.

### Mistake 4: Forgetting model.eval()

Dropout and BatchNorm behave differently during evaluation.

**Solution:** Call `model.eval()` before evaluation or prediction.

### Mistake 5: Forgetting torch.no_grad()

PyTorch may unnecessarily track gradients during evaluation.

**Solution:** Use `torch.no_grad()` for ordinary inference.

### Mistake 6: Using different transforms during prediction

If training images were normalized but prediction images are not, the model receives inputs with a different scale.

**Solution:** Reuse the same normalization procedure.

### Mistake 7: Changing architecture after saving

The saved state dictionary may no longer match the architecture.

**Solution:** Keep the training and prediction architectures identical.

### Mistake 8: Calling scheduler.step() in the batch loop

This can reduce the learning rate much more frequently than intended.

**Solution:** For our StepLR configuration, call it once after each training epoch.

---

# 22. What We Have Learned

After implementing and experimenting with CNNs, we have covered:

- Image tensors and channel dimensions
- Convolution and learnable filters
- Feature maps
- ReLU activation
- Max pooling
- Flattening and linear layers
- Tensor shape calculation
- Logits and class predictions
- Cross-entropy loss
- Forward propagation
- Backpropagation
- Adam optimizer
- Training accuracy and loss
- Evaluation mode and gradient control
- Saving and loading model parameters
- Image normalization
- Data augmentation
- Dropout
- Batch Normalization
- Learning rate scheduling
- Accuracy, precision, recall and F1-score
- Confusion matrices
- Experimental comparison and generalization

---

# 23. Practical Understanding Checklist

Before moving forward, make sure you can do these without simply copying the original code.

- [ ] Explain why CNNs are suitable for image data.
- [ ] Explain how a 3 × 3 convolution filter works.
- [ ] Calculate convolution output dimensions.
- [ ] Explain why ReLU is used.
- [ ] Calculate the output shape after MaxPool2d.
- [ ] Calculate the input size required by a Linear layer.
- [ ] Explain logits and CrossEntropyLoss.
- [ ] Explain forward pass and backpropagation.
- [ ] Explain why gradients must be cleared.
- [ ] Explain how Adam updates parameters.
- [ ] Explain the difference between `train()` and `eval()`.
- [ ] Explain the difference between `eval()` and `no_grad()`.
- [ ] Save and load a trained model.
- [ ] Explain why augmentation is applied only to training images.
- [ ] Explain Dropout and BatchNorm.
- [ ] Explain how StepLR changes the learning rate.
- [ ] Interpret a confusion matrix and classification report.
- [ ] Build a small CNN independently and test it on new images.

---

## Final Takeaway

Building a CNN is not just about stacking convolutional layers.

A successful image classification workflow requires understanding:

1. How images are represented as tensors.
2. How convolutional layers extract features.
3. How the model converts features into class scores.
4. How loss and backpropagation train the model.
5. How optimization techniques affect learning.
6. How evaluation tells us whether the model generalizes to unseen data.

Our best CIFAR-10 result so far is **72.18% test accuracy**, using data augmentation and Batch Normalization.

That is a useful baseline, not the limit of what CNNs can achieve. The next goal is to understand and build stronger image models while keeping the experimental process reliable.