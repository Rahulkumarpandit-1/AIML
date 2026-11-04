# 12. Transfer Learning and Data Augmentation

## 1. What is Transfer Learning?

Transfer Learning is a technique where we take a model already trained on a large dataset and reuse its learned knowledge for a different task.

For example:

- ResNet18 was pretrained on ImageNet.
- ImageNet contains 1,000 object categories.
- We want to classify CIFAR-10 images into 10 categories.
- Instead of training a CNN from scratch, we reuse ResNet18's pretrained weights.

### Basic workflow

1. Load a pretrained model.
2. Replace its original classification layer.
3. Freeze pretrained layers.
4. Train the new classification layer.
5. Optionally unfreeze some pretrained layers.
6. Fine-tune the model on the new dataset.

## 2. Pretrained Weights

Pretrained weights are parameters (weights and biases) learned during previous training.

We load them using:

```python
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

The model already knows useful visual patterns, such as:

- Edges
- Textures
- Shapes
- Object parts
- Complex visual patterns

We reuse this knowledge instead of starting with random weights.

## 3. Feature Extraction

Feature extraction means keeping the pretrained feature extractor frozen and training only a new classification layer.

### Step 1: Replace the classifier

```python
import torch.nn as nn

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

The original ResNet18 classifier produces 1,000 outputs because ImageNet has 1,000 classes.

CIFAR-10 has only 10 classes, so we replace it with a classifier that produces 10 outputs.

### Step 2: Freeze pretrained parameters

```python
for param in model.parameters():
    param.requires_grad = False
```

This prevents the pretrained parameters from receiving gradients.

### Step 3: Unfreeze the new classifier

```python
for param in model.fc.parameters():
    param.requires_grad = True
```

Now only the new classifier is trainable.

### Important distinction

- Frozen parameters: Their values are not updated through gradient-based optimization.
- Trainable parameters: Their gradients can be calculated and their values updated by the optimizer.

## 4. Understanding ResNet18 Architecture

ResNet18 is a convolutional neural network containing residual connections.

Its major components are:

| Component | Channels | Residual blocks |
|---|---:|---:|
| Conv1 | 64 | - |
| Layer 1 | 64 | 2 |
| Layer 2 | 128 | 2 |
| Layer 3 | 256 | 2 |
| Layer 4 | 512 | 2 |
| FC | 10 (our task) | - |

Each residual block contains convolution operations, Batch Normalization and ReLU, along with a skip connection.

### What do the four stages learn?

- Layer 1: Basic edges and textures.
- Layer 2: Corners and simple patterns.
- Layer 3: More complex patterns and object parts.
- Layer 4: Higher-level visual representations.

These are intuitive descriptions. Features are learned across the network rather than in completely separate categories.

### Understanding layer4

In our experiment, we unfreeze layer4 because its high-level features can adapt to the new dataset.

We keep the earlier layers frozen because their general visual features are already useful.

## 5. Fine-Tuning

Fine-tuning means unfreezing some pretrained layers and allowing their parameters to adapt to the new task.

In our experiment, we unfreeze layer4 and the new classifier.

```python
# Unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# Unfreeze classifier
for param in model.fc.parameters():
    param.requires_grad = True
```

Layers 1, 2 and 3 remain frozen.

### Why use different learning rates?

Pretrained layers already contain useful knowledge, so we make smaller updates to them.

The new classifier starts with randomly initialized parameters, so it can use a larger learning rate.

```python
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])
```

| Component | Learning rate |
|---|---:|
| Layer 4 | 0.00001 |
| FC classifier | 0.0001 |

## 6. Batch Normalization During Fine-Tuning

Batch Normalization maintains running statistics used to normalize activations.

During training, BatchNorm normally updates these statistics.

For our experiment, we wanted to keep the pretrained running statistics fixed.

```python
model.train()

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
```

This prevents BatchNorm running statistics from updating.

Important: Calling `eval()` on BatchNorm does not automatically freeze its learnable weight and bias parameters. Their gradient behavior is controlled separately by `requires_grad`.

## 7. Data Augmentation

Data augmentation creates modified versions of training images to increase variation in the training data.

Common transformations include:

- Random horizontal flipping
- Random cropping
- Rotation
- Color variation
- Resizing

Example:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
```

Augmentation helps reduce overfitting by exposing the model to different versions of training images.

Important:
- Apply random augmentation to training data.
- Keep test transformations deterministic.
- Use normalization appropriate for the pretrained model when using pretrained weights.

## 8. Pretrained Model Transforms

For ResNet18, we used:

```python
transform = weights.transforms()
```

This applies the preprocessing expected by the pretrained model, including resizing, cropping and ImageNet normalization.

CIFAR-10 images originally have dimensions of 32 × 32 pixels.

The pretrained ResNet18 transform converts them to 224 × 224 pixels.

This allows us to use the pretrained model's expected input format.

Note: `weights.transforms()` is preprocessing, not random data augmentation.

## 9. Our CIFAR-10 Transfer Learning Experiment

Dataset:

- Training images: 50,000
- Testing images: 10,000
- Classes: 10
- Batch size: 128

The ten classes are:

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

### Training workflow

```python
for epoch in range(epochs):

    model.train()

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, labels)

        loss.backward()

        optimizer.step()
```

### What each operation does

| Operation | Purpose |
|---|---|
| `model.train()` | Enables training behavior |
| `optimizer.zero_grad()` | Clears previous gradients |
| `model(images)` | Forward propagation |
| `loss_fn(logits, labels)` | Calculates classification loss |
| `loss.backward()` | Calculates gradients |
| `optimizer.step()` | Updates trainable parameters |

### Testing

```python
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
```

`model.eval()` changes the behavior of layers such as BatchNorm and Dropout.

`torch.no_grad()` disables gradient tracking and avoids building the backward computation graph.

They serve different purposes.

## 10. Experimental Results

### Experiment 1: Feature Extraction

Only the new FC classifier was trained.

| Metric | Result |
|---|---:|
| Final training accuracy | 76.57% |
| Test accuracy | 76.46% |

### Experiment 2: Fine-Tuning

Layer4 and the FC classifier were trained.

| Epoch | Training Loss | Training Accuracy |
|---|---:|---:|
| 1 | 0.6830 | 77.10% |
| 2 | 0.3329 | 88.76% |
| 3 | 0.2720 | 90.64% |
| 4 | 0.2272 | 92.37% |
| 5 | 0.1955 | 93.41% |

Final test accuracy: **90.50%**

### Comparison

| Method | Test Accuracy |
|---|---:|
| Feature Extraction | 76.46% |
| Fine-Tuning | 90.50% |

Improvement: 14.04 percentage points.

This experiment demonstrates that adapting pretrained high-level features can improve performance on a new classification task.

These results are specific to our experiment and are not guaranteed for every dataset or training configuration.

## 11. Important Concepts to Remember

1. Transfer learning reuses knowledge from a pretrained model.
2. Feature extraction freezes pretrained layers and trains a new classifier.
3. Fine-tuning unfreezes some pretrained layers.
4. ResNet18 contains four major feature-extraction stages.
5. Layer4 contains higher-level visual representations.
6. Different learning rates can be used for different parameter groups.
7. Data augmentation increases training-image variation.
8. `model.eval()` and `torch.no_grad()` have different purposes.
9. Test accuracy measures performance on held-out data.
10. Fine-tuning can improve accuracy, but improvement must be verified experimentally.

## 12. Practical Learning Status

- [x] Loaded pretrained ResNet18.
- [x] Replaced the original classifier.
- [x] Implemented feature extraction.
- [x] Implemented partial fine-tuning.
- [x] Used different learning rates.
- [x] Evaluated both experiments on CIFAR-10.
- [ ] Independently implement transfer learning without following a tutorial.
