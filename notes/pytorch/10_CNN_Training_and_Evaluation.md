# 10 — CNN Training and Evaluation

## 1. CIFAR-10 Image Classification Project

We used CIFAR-10 to implement and train a CNN using PyTorch.

CIFAR-10 contains:

- 50,000 training images.
- 10,000 test images.
- 10 classes.
- RGB images of size 32 × 32.

The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

Each image has the shape:

`[3, 32, 32]`

A batch of 64 images has the shape:

`[64, 3, 32, 32]`

## 2. Dataset and DataLoader

A Dataset provides individual samples and their labels.

A DataLoader loads samples in batches and can shuffle the training data.

Example:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
```

### Why shuffle training data?

Shuffling changes the order in which training examples are presented.

It helps avoid learning patterns that depend on a fixed dataset order.

For evaluation, shuffling is not necessary. Keeping a consistent order can make debugging and analysis easier.

### What is an epoch?

An epoch is one complete pass through the training dataset.

With 50,000 images and batch size 64:

\[
\text{Batches per epoch}
=
\left\lceil\frac{50000}{64}\right\rceil
=782
\]

The model processes all batches before completing one epoch.

## 3. Understanding Logits

The final Linear layer produces ten values for each CIFAR-10 image.

These values are called logits.

Example:

```text
Airplane:     2.1
Automobile:   0.5
Bird:        -1.2
Cat:          0.8
Deer:         0.2
Dog:          1.4
Frog:        -0.5
Horse:        0.3
Ship:         0.9
Truck:        0.1
```

The largest logit is the predicted class.

PyTorch:

```python
predictions = logits.argmax(dim=1)
```

For a batch of images, `dim=1` selects the highest-scoring class for each image.

### Logits versus probabilities

Logits are raw scores, not probabilities.

Softmax converts logits into values between 0 and 1 that sum to 1.

```python
probabilities = torch.softmax(logits, dim=1)
```

Softmax is useful when displaying class probabilities.

However, we do not need to apply softmax before `nn.CrossEntropyLoss`.

## 4. CrossEntropyLoss

For a classification model:

```python
loss_fn = nn.CrossEntropyLoss()
```

CrossEntropyLoss measures how far the model's predictions are from the correct class labels.

Suppose:

- Actual class: Cat
- Model predicts: Dog

The model should receive a higher loss because its prediction is incorrect.

If the model assigns a high probability to the correct class, the loss is generally lower.

The training code is:

```python
loss = loss_fn(logits, labels)
```

Where:

- `logits`: Raw output scores from the model.
- `labels`: Correct class indices.

### Important

Correct:

```python
loss = loss_fn(logits, labels)
```

Incorrect for this loss function:

```python
probabilities = torch.softmax(logits, dim=1)
loss = loss_fn(probabilities, labels)
```

`CrossEntropyLoss` internally handles the log-softmax operation.

## 5. Forward Pass

The forward pass means sending input data through the network to obtain predictions.

```python
logits = model(images)
```

The process:

1. Images enter the CNN.
2. Convolution layers extract features.
3. Activation functions introduce non-linearity.
4. Pooling reduces spatial dimensions.
5. Flatten converts feature maps into vectors.
6. Linear produces class scores.

The output is the logits tensor.

For a batch of 64 CIFAR-10 images:

`[64, 3, 32, 32] → [64, 10]`

## 6. Backpropagation

Backpropagation calculates gradients of the loss with respect to the trainable model parameters.

These gradients tell the optimizer how the parameters contributed to the error.

The process uses the chain rule of differentiation.

PyTorch:

```python
loss.backward()
```

This calculates gradients and stores them in the parameters' `.grad` attributes.

Backpropagation does not itself update the model's weights. The optimizer performs the update.

## 7. Optimizer and Adam

An optimizer updates model parameters using gradients.

We used Adam:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam maintains moving estimates of gradients and squared gradients to adapt parameter updates.

### Learning rate

The learning rate controls the general scale of parameter updates.

- Too high: Training can become unstable.
- Too low: Training can become very slow.
- Appropriate: The model can learn effectively.

The learning rate is a hyperparameter. It is not automatically learned as an ordinary model weight.

## 8. Complete Training Step

```python
optimizer.zero_grad()

logits = model(images)

loss = loss_fn(logits, labels)

loss.backward()

optimizer.step()
```

Understand each operation:

| Operation | Purpose |
|---|---|
| `optimizer.zero_grad()` | Clears previous gradients |
| `model(images)` | Forward pass |
| `loss_fn(...)` | Calculates prediction error |
| `loss.backward()` | Calculates gradients |
| `optimizer.step()` | Updates model weights |

### Why clear gradients?

PyTorch accumulates gradients by default.

Without clearing them, gradients from the previous batch would be added to the current batch's gradients.

Therefore, we normally call `optimizer.zero_grad()` before every training batch.

## 9. Training Loop

A simplified training loop:

```python
for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={train_accuracy:.2f}%"
    )
```

### Understanding training accuracy

Training accuracy measures how many training images the model predicts correctly.

\[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\times100
\]

For example, if 640 out of 1,000 images are classified correctly:

\[
Accuracy = \frac{640}{1000}\times100=64\%
\]

Training accuracy helps us understand how well the model fits the training data.

It does not tell us by itself how well the model generalizes to unseen images.

## 10. Training Loss and Accuracy

These metrics measure different things.

### Loss

Measures prediction error according to the loss function.

Lower loss generally indicates better fitting to the training data.

### Accuracy

Measures the percentage of correct predictions.

A model can have decreasing loss without a large accuracy increase because loss also reflects how confident the model is.

For example, the model may predict the correct class with increasing confidence while its number of correct predictions remains unchanged.

Therefore, we monitor both.

## 11. Training Mode and Evaluation Mode

PyTorch provides:

```python
model.train()
```

and:

```python
model.eval()
```

### Training mode

`model.train()` enables training-specific behaviour in layers such as:

- Dropout
- Batch Normalization

### Evaluation mode

`model.eval()` changes the behaviour of those layers:

- Dropout is disabled.
- BatchNorm uses stored running statistics.

### Difference between eval() and no_grad()

These operations are different.

```python
model.eval()

with torch.no_grad():
    logits = model(images)
```

`model.eval()` changes the behaviour of certain layers.

`torch.no_grad()` disables gradient tracking.

For ordinary evaluation and inference, we generally use both.

## 12. Test Evaluation

A basic evaluation loop:

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

### Understanding the code

`model.eval()`

Switches the model to evaluation mode.

`torch.no_grad()`

Prevents gradient tracking.

`logits = model(images)`

Obtains model outputs.

`argmax(dim=1)`

Selects the highest-scoring class.

`predictions == labels`

Compares predictions with the actual labels.

`sum().item()`

Counts correct predictions and converts the result to a Python number.

## 13. Classification Metrics

Accuracy alone does not explain every aspect of model performance.

We also use precision, recall, F1-score and confusion matrices.

### 13.1 Accuracy

\[
Accuracy =
\frac{TP+TN}{TP+TN+FP+FN}
\]

For multiclass classification, accuracy is the total number of correct predictions divided by the total number of predictions.

### 13.2 Precision

Precision answers:

**When the model predicts a particular class, how often is it correct?**

\[
Precision = \frac{TP}{TP+FP}
\]

Example:

The model predicts 100 images as cats. Only 70 are actually cats.

Precision = 70%.

### 13.3 Recall

Recall answers:

**Out of all actual examples of a class, how many did the model identify correctly?**

\[
Recall = \frac{TP}{TP+FN}
\]

Example:

There are 100 actual cat images. The model correctly identifies 60.

Recall = 60%.

### 13.4 F1-score

F1-score is the harmonic mean of precision and recall.

\[
F1 =
2\times
\frac{Precision\times Recall}
{Precision+Recall}
\]

It balances precision and recall.

### 13.5 Confusion Matrix

A confusion matrix compares actual and predicted classes.

For CIFAR-10, it has ten rows and ten columns.

- Rows: Actual classes.
- Columns: Predicted classes.
- Diagonal: Correct predictions.
- Off-diagonal: Incorrect predictions.

It helps identify which classes the model confuses.

For example, a model may confuse cats and dogs because their visual patterns can be similar.

## 14. Classification Report

Scikit-learn can generate a classification report:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
```

The report provides precision, recall, F1-score and support for each class.

### What is support?

Support is the number of actual examples belonging to a class in the evaluated dataset.

For CIFAR-10, the standard test set contains 1,000 images per class.

### Macro average

Calculates the unweighted average of a metric across classes.

Each class receives equal importance.

### Weighted average

Calculates the average weighted by each class's support.

For a balanced CIFAR-10 test set, macro and weighted averages are often similar.

## 15. Saving Model Parameters

After training:

```python
torch.save(
    model.state_dict(),
    "cnn_model.pth"
)
```

A `state_dict` stores learned parameters and relevant buffers.

It does not automatically store the complete Python architecture.

## 16. Loading a Model

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

Then load its parameters:

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

The saved parameters correspond to particular layers and tensor dimensions.

If the architecture changes, parameter shapes or state dictionary keys may not match.

For example, changing the first convolution from 16 output channels to 32 changes its weight dimensions and the dimensions of later layers.

The training and prediction scripts must therefore use compatible architectures.

## 17. Making Predictions on an Image

Once the model is loaded:

```python
model.eval()

with torch.no_grad():

    image = image.to(device)

    logits = model(image.unsqueeze(0))

    prediction = logits.argmax(dim=1).item()

    print(class_names[prediction])
```

`unsqueeze(0)` adds the batch dimension.

A single image has shape:

`[3, 32, 32]`

The model expects:

`[1, 3, 32, 32]`

Therefore, we add a batch dimension before prediction.

### Important

The prediction image must use the same essential preprocessing as the training images, especially normalization.

## 18. Complete CNN Workflow

```text
CIFAR-10 Dataset
       |
       v
Dataset and DataLoader
       |
       v
Image Preprocessing
       |
       v
Build CNN
       |
       v
Initialize Loss and Optimizer
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
Optimizer Updates Weights
       |
       v
Repeat Across Batches and Epochs
       |
       v
Evaluate Model
       |
       v
Calculate Metrics
       |
       v
Save Model Parameters
       |
       v
Load Model
       |
       v
Predict New Images
```

## 19. Common Mistakes

1. Applying softmax before CrossEntropyLoss.
2. Forgetting `optimizer.zero_grad()`.
3. Forgetting `model.eval()` during evaluation.
4. Forgetting `torch.no_grad()` during ordinary inference.
5. Using different normalization during prediction.
6. Forgetting the batch dimension for a single image.
7. Loading a state dictionary into an incompatible architecture.
8. Using training accuracy as the only measure of model quality.

## 20. Practical Understanding Checklist

- [ ] Explain logits and probabilities.
- [ ] Explain CrossEntropyLoss.
- [ ] Explain the forward pass.
- [ ] Explain backpropagation.
- [ ] Explain the role of Adam.
- [ ] Explain the purpose of the learning rate.
- [ ] Write a training loop independently.
- [ ] Calculate training accuracy.
- [ ] Explain `model.train()` and `model.eval()`.
- [ ] Explain `torch.no_grad()`.
- [ ] Evaluate a model on test data.
- [ ] Interpret precision, recall and F1-score.
- [ ] Interpret a confusion matrix.
- [ ] Save and load a trained CNN.
- [ ] Predict the class of a new image.

**Final takeaway:** Training teaches the CNN to recognize patterns. Evaluation measures whether those learned patterns work on images that were not used for gradient updates.