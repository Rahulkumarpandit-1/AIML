# 11 — CNN Experiments and Improvements

## 1. Why Improve a CNN?

Building a CNN is only the beginning.

After training a baseline model, we need to evaluate its performance and identify possible improvements.

A model may perform well on training data but struggle on unseen images. This is called poor generalization.

We experimented with several techniques:

1. Data normalization
2. Data augmentation
3. Dropout
4. Batch Normalization
5. Learning rate scheduling

The objective was to understand how these techniques affect model performance.

---

# 2. Data Normalization

Normalization changes the scale of input data.

We used:

```python
transforms.ToTensor()
```

This converts image data into tensors and typically scales 8-bit pixel values from `[0, 255]` to `[0, 1]`.

Then:

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

Normalization performs:

\[
x_{\text{normalized}} =
\frac{x-\text{mean}}{\text{standard deviation}}
\]

For our values:

\[
x_{\text{normalized}}=\frac{x-0.5}{0.5}
\]

Therefore, values in `[0, 1]` become approximately `[-1, 1]`.

### Why normalize?

- Keeps input scales consistent.
- Can make optimization easier.
- Often helps neural networks train more reliably.

### Important

Use consistent normalization during training, testing and prediction.

If the model was trained using normalized images, feeding it unnormalized images during prediction changes the input distribution.

---

# 3. Data Augmentation

Data augmentation creates varied versions of training images using transformations.

It helps the model learn useful patterns instead of memorizing the exact training images.

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

## 3.1 RandomCrop

```python
transforms.RandomCrop(32, padding=4)
```

This adds padding around the image and then randomly crops it back to 32 × 32.

The object may appear at slightly different positions.

This helps the model become less sensitive to small changes in object position.

## 3.2 RandomHorizontalFlip

```python
transforms.RandomHorizontalFlip()
```

Randomly flips images horizontally.

For example, an automobile facing left may be shown facing right.

This gives the model more variation during training.

## 3.3 Why augment only training images?

Training augmentation helps the model learn from varied examples.

The test set should provide a consistent evaluation of generalization.

Therefore:

| Dataset | Transformations |
|---|---|
| Training | RandomCrop, RandomHorizontalFlip, ToTensor, Normalize |
| Testing | ToTensor, Normalize |

Random augmentation is generally not applied to the test set during ordinary evaluation.

## 3.4 Why can augmentation improve generalization?

Without augmentation, the model sees only the original training examples.

With augmentation, the same image can appear in slightly different forms across epochs.

This encourages the model to learn features that remain useful under those transformations.

### Our experiment

| Configuration | Test accuracy |
|---|---:|
| Original CNN | 68.16% |
| CNN + Data augmentation | 71.17% |

Improvement:

`71.17 - 68.16 = 3.01 percentage points`

This was a meaningful improvement in our experiment.

---

# 4. Dropout

Dropout is a regularization technique.

It randomly sets some activations to zero during training.

We experimented with:

```python
nn.Flatten(),
nn.Dropout(p=0.1),
nn.Linear(32 * 8 * 8, 10)
```

## 4.1 Understanding the dropout probability

For:

```python
nn.Dropout(p=0.1)
```

approximately 10% of the selected activations are randomly set to zero during each training forward pass.

The remaining activations are scaled appropriately by PyTorch during training.

The dropped activations are not permanently deleted.

Different activations can be dropped in different batches.

## 4.2 Why use Dropout?

Without sufficient regularization, a model may depend too heavily on particular activations.

Dropout discourages this behaviour.

It can help reduce overfitting by encouraging the model to learn more robust representations.

## 4.3 Dropout during evaluation

When we call:

```python
model.eval()
```

Dropout is disabled.

The complete set of activations is used.

## 4.4 Our experiment

The Dropout p=0.1 run achieved:

**Test accuracy: 69.02%**

This was lower than our augmentation-only and augmentation-plus-BatchNorm results.

This does not mean Dropout is always harmful.

Its effect depends on:

- Dropout probability.
- Model architecture.
- Dataset size.
- Training duration.
- Other regularization techniques.

A technique should be judged by controlled experiments, not by its reputation.

---

# 5. Batch Normalization

Batch Normalization normalizes activations within a neural network.

For convolutional layers, PyTorch provides:

```python
nn.BatchNorm2d(number_of_channels)
```

We added it after convolution layers.

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

## 5.1 What does BatchNorm do?

During training, BatchNorm uses the current mini-batch's mean and variance to normalize activations.

It also learns two parameters:

- Gamma: Controls scaling.
- Beta: Controls shifting.

This allows the network to adjust the normalized activations.

## 5.2 Why BatchNorm2d(16)?

The first convolution produces 16 output channels.

Therefore, we use:

```python
nn.BatchNorm2d(16)
```

The second convolution produces 32 output channels.

Therefore:

```python
nn.BatchNorm2d(32)
```

The number passed to BatchNorm2d must match the number of channels being normalized.

## 5.3 Why place BatchNorm after Conv2d?

Our layer sequence is:

```text
Convolution
     |
     v
Batch Normalization
     |
     v
ReLU
     |
     v
Max Pooling
```

The convolution produces feature maps.

BatchNorm normalizes their activations.

ReLU then applies the non-linear activation.

## 5.4 BatchNorm during training and evaluation

During training:

- Uses current batch statistics.
- Updates running mean and variance.

During evaluation:

- Uses stored running statistics.
- Does not use the current evaluation batch statistics for ordinary inference.

This is why `model.eval()` is important.

## 5.5 Our experiment

| Configuration | Test accuracy |
|---|---:|
| Original CNN | 68.16% |
| Augmentation | 71.17% |
| Augmentation + BatchNorm | **72.18%** |

This was our best result.

Improvement over the original CNN:

`72.18 - 68.16 = 4.02 percentage points`

BatchNorm improved the result in our experiment, but this does not guarantee an improvement in every model.

---

# 6. Learning Rate Scheduling

The learning rate controls how large the optimizer's updates are.

We initially used:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

A fixed learning rate keeps the same learning rate throughout training.

A scheduler changes it according to a defined strategy.

## 6.1 StepLR

We used:

```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)
```

Parameters:

| Parameter | Meaning |
|---|---|
| `step_size=5` | Reduces the learning rate every five scheduler steps |
| `gamma=0.5` | Multiplies the current learning rate by 0.5 |

Starting learning rate:

`0.001`

The schedule:

| Epochs | Learning rate used |
|---|---:|
| 1–5 | 0.001 |
| 6–10 | 0.0005 |
| 11–15 | 0.00025 |
| After epoch 15 | 0.000125 |

## 6.2 Why reduce the learning rate?

At the beginning of training, larger updates may help the model learn quickly.

Later, smaller updates may help refine the learned parameters.

A scheduler provides a controlled way to change the learning rate during training.

## 6.3 Where to call scheduler.step()?

For our StepLR setup, call it once after all training batches in an epoch.

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

Do not put `scheduler.step()` inside the batch loop.

## 6.4 Important detail about printed learning rates

Suppose epoch 5 trains using learning rate 0.001.

At the end of epoch 5, `scheduler.step()` changes the learning rate to 0.0005.

If we print the learning rate after calling the scheduler, the printed value represents the next epoch's learning rate.

This explains why the training log may show 0.000500 on the epoch 5 line even though epoch 5 was trained at 0.001.

## 6.5 Our StepLR experiment

The final results were:

- Epochs: 15
- Final training loss: 0.9508
- Training accuracy: 67.19%
- Test accuracy: 71.06%
- Initial learning rate: 0.001

StepLR did not improve test accuracy over our augmentation-plus-BatchNorm configuration.

A learning rate scheduler is not automatically beneficial. Its settings must be tested.

---

# 7. Experiment Results

We compared the following configurations:

| Experiment | Test accuracy |
|---|---:|
| Original CNN | 68.16% |
| Data augmentation | 71.17% |
| Augmentation + BatchNorm | **72.18%** |
| Augmentation + BatchNorm + StepLR | 71.06% |
| Augmentation + Dropout (p=0.1) | 69.02% |

## 7.1 Best-performing configuration

Our best configuration was:

**Data Augmentation + Batch Normalization**

Test accuracy:

**72.18%**

Improvement over the baseline:

**4.02 percentage points**

## 7.2 What did the experiments teach us?

### Data augmentation

Improved generalization in our experiment by exposing the model to varied training images.

### Dropout

Did not improve our result with p=0.1.

This suggests that adding regularization without considering the rest of the training setup may not help.

### BatchNorm

Produced our best test accuracy.

It changed the activation behaviour and improved the result in this particular model.

### StepLR

Changed the learning rate over time but did not beat the BatchNorm-only configuration.

A scheduler is a tool for controlling optimization, not a guaranteed accuracy improvement.

---

# 8. Training Accuracy vs Test Accuracy

Training accuracy and test accuracy measure performance on different datasets.

Training accuracy:

- Calculated on training images.
- Measures how well the model fits its training data.

Test accuracy:

- Calculated on test images.
- Measures performance on images not used for gradient updates.

Sometimes test accuracy can be higher than training accuracy.

Reasons include:

- Training augmentation makes the training images harder.
- Dropout is active during training.
- BatchNorm behaves differently in training and evaluation modes.
- Training accuracy is measured while the model is being updated.

Therefore, a higher test accuracy does not automatically indicate a bug.

However, unexpected differences should still be investigated.

---

# 9. Avoiding Overfitting

Overfitting occurs when a model learns the training data too specifically and performs poorly on unseen data.

Common signs:

- Training accuracy keeps increasing.
- Validation accuracy stops improving or decreases.
- Training loss decreases while validation loss increases.

Ways to reduce overfitting include:

1. Data augmentation.
2. Dropout.
3. Weight decay.
4. Batch Normalization where appropriate.
5. Early stopping.
6. More representative training data.
7. Appropriate model complexity.

No single technique is guaranteed to work in every situation.

## Validation vs Test Data

These datasets have different purposes.

**Training set:** Used to update model parameters.

**Validation set:** Used to compare configurations and tune hyperparameters.

**Test set:** Used for final evaluation after model decisions are made.

Ideally, we should select Dropout probability, scheduler settings and other hyperparameters using validation data.

Repeatedly selecting configurations based on test accuracy makes the test set less independent as a final evaluation.

Our experiment results are useful for learning, but a more rigorous comparison should use a fixed validation split and reserve the test set for the final evaluation.

---

# 10. How to Conduct a Fair Experiment

When comparing CNN configurations:

1. Use the same training and validation split.
2. Keep the number of epochs consistent when appropriate.
3. Keep the batch size consistent.
4. Keep the evaluation procedure consistent.
5. Record the random seed.
6. Change one major factor at a time.
7. Compare validation metrics.
8. Evaluate the selected configuration on the test set.

Record:

- Model architecture.
- Optimizer.
- Initial learning rate.
- Scheduler configuration.
- Batch size.
- Number of epochs.
- Training loss.
- Training accuracy.
- Validation accuracy.
- Final test accuracy.

This makes experiments easier to reproduce and interpret.

---

# 11. Important Comparison Table

| Technique | Main purpose | Possible benefit |
|---|---|---|
| Normalization | Standardizes input scale | More consistent optimization |
| Data augmentation | Creates varied training examples | Better generalization |
| Dropout | Randomly disables activations | Can reduce overfitting |
| BatchNorm | Normalizes activations | Can stabilize and improve training |
| StepLR | Changes learning rate | Can refine optimization |

These techniques solve different problems. They are not interchangeable.

---

# 12. Practical Understanding Checklist

- [ ] Explain why image normalization is useful.
- [ ] Explain RandomCrop and RandomHorizontalFlip.
- [ ] Explain why augmentation is applied to training images.
- [ ] Explain how Dropout works during training.
- [ ] Explain why Dropout is disabled during evaluation.
- [ ] Explain BatchNorm's training and evaluation behaviour.
- [ ] Explain why BatchNorm2d needs the number of channels.
- [ ] Explain StepLR's step_size and gamma.
- [ ] Place scheduler.step() correctly.
- [ ] Interpret the learning rate in training logs.
- [ ] Compare training and test accuracy.
- [ ] Explain overfitting.
- [ ] Distinguish training, validation and test data.
- [ ] Design a fair experiment.
- [ ] Explain why the best result is not necessarily universal.

---

## Final Takeaway

Improving a CNN is an experimental process.

We should not add techniques simply because they are popular.

Instead:

1. Establish a baseline.
2. Identify a problem.
3. Choose a technique that addresses it.
4. Run a controlled experiment.
5. Compare validation performance.
6. Keep changes that genuinely improve the model.

Our best result so far is **72.18% test accuracy**, achieved with data augmentation and Batch Normalization.

Treat this as our current experimental baseline, not the final limit of the model.