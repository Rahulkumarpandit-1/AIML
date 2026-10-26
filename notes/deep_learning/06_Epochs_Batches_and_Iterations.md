# Epochs, Batches, and Iterations

## 1. Batch

A batch is a subset of training examples processed together.

Example: If the batch size is 100, the model processes 100 examples in each training step.

## 2. Iteration

An iteration is one training step. In a typical mini-batch training loop, the model processes a batch, calculates the loss and gradients, and the optimizer updates the parameters once.

## 3. Epoch

An epoch is one complete pass through the entire training dataset.

If the dataset contains 1,200 examples and the batch size is 100, one epoch contains 12 batches and typically 12 optimizer updates.

## 4. Calculate Iterations

When the dataset is processed in batches, including a potentially smaller final batch:

`iterations_per_epoch = ceil(dataset_size / batch_size)`

Example:

`dataset_size = 1200`
`batch_size = 100`

`iterations_per_epoch = 1200 / 100 = 12`

For 5 epochs:

`total_iterations = 12 * 5 = 60`

## 5. Effect of Batch Size

For a fixed dataset:

- **Smaller batch:** More iterations per epoch; often less memory per batch.
- **Larger batch:** Fewer iterations per epoch; often more memory per batch.
- Neither is always best. The choice depends on the model, hardware, data, and optimization behavior.

## 6. Important Distinction

- Batch = a group of examples.
- Iteration = one training step, usually one optimizer update.
- Epoch = one complete pass through the training dataset.

**Remember:** `iterations per epoch ≈ dataset size / batch size`, rounded up when the division is not exact.