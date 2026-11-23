# Object Detection and Segmentation

## 1. Computer Vision Tasks

Computer Vision allows machines to understand and interpret visual information.

Three important tasks:

- **Image Classification:** Predicts what an image contains.
- **Object Detection:** Identifies objects and locates them using bounding boxes.
- **Image Segmentation:** Identifies the pixels belonging to objects.

### Types of Segmentation

**Semantic Segmentation**
- Assigns a class to every pixel.
- Objects belonging to the same class share the same label.
- Does not distinguish individual objects of the same class.

**Instance Segmentation**
- Identifies each individual object separately.
- Gives every detected object its own pixel-level mask.
- Example: Two pedestrians receive two separate masks.

## 2. Bounding Boxes

A bounding box represents an object's location using four coordinates:

`[x_min, y_min, x_max, y_max]`

- `x_min`: Left boundary.
- `y_min`: Top boundary.
- `x_max`: Right boundary.
- `y_max`: Bottom boundary.

For object detection, the model predicts bounding boxes and class labels.

## 3. Intersection over Union (IoU)

IoU measures the overlap between a predicted region and the actual ground-truth region.

`IoU = Intersection Area / Union Area`

- IoU = 1: Perfect overlap.
- IoU = 0: No overlap.
- Higher IoU means better alignment.

IoU is used in both bounding-box and segmentation evaluation.

## 4. Non-Maximum Suppression (NMS)

Object detectors may generate multiple overlapping predictions for the same object.

NMS removes redundant predictions.

Basic process:

1. Select the prediction with the highest confidence.
2. Compare other predictions with it using IoU.
3. Remove predictions whose overlap exceeds the NMS threshold.
4. Repeat with the remaining predictions.

## 5. Faster R-CNN

Faster R-CNN is a two-stage object detection architecture.

### Stage 1: Region Proposal Network (RPN)

- Examines feature maps.
- Proposes regions that may contain objects.
- Predicts objectness scores and bounding-box adjustments.

### Stage 2: RoI Heads

- Examines proposed regions.
- Classifies objects.
- Refines bounding boxes.

### Important Components

- Backbone: Extracts image features.
- Feature Pyramid Network (FPN): Combines features at different scales.
- RPN: Generates object proposals.
- RoI Heads: Classifies and refines proposed objects.

### Faster R-CNN Losses

- `loss_classifier`: Object classification error.
- `loss_box_reg`: Bounding-box regression error.
- `loss_objectness`: RPN objectness error.
- `loss_rpn_box_reg`: RPN box regression error.

## 6. YOLO vs Faster R-CNN

**YOLO**
- One-stage detector.
- Predicts objects in a single detection pipeline.
- Commonly used when fast inference is important.

**Faster R-CNN**
- Two-stage detector.
- First proposes regions, then classifies and refines them.
- Often useful when detection quality is a priority.

## 7. Mask R-CNN

Mask R-CNN extends Faster R-CNN by adding a mask prediction branch.

It performs three tasks:

1. Object classification.
2. Bounding-box prediction.
3. Pixel-level mask prediction.

### Architecture

- Backbone: Extracts image features.
- FPN: Provides multi-scale features.
- RPN: Generates object proposals.
- RoI Heads: Predicts class labels and bounding boxes.
- Mask Head: Predicts an individual mask for each detected object.

### Mask R-CNN Losses

- `loss_classifier`: Classification error.
- `loss_box_reg`: Bounding-box regression error.
- `loss_mask`: Pixel-level mask prediction error.
- `loss_objectness`: RPN objectness error.
- `loss_rpn_box_reg`: RPN box regression error.

Total loss is the sum of the individual losses.

## 8. Penn-Fudan Pedestrian Dataset

Used for practical Faster R-CNN and Mask R-CNN implementation.

- 170 images.
- 345 pedestrians.
- RGB images stored in `PNGImages`.
- Instance masks stored in `PedMasks`.

### Dataset Preparation

Each nonzero mask ID represents a separate pedestrian.

- Pixel value 0: Background.
- Nonzero pixel values: Individual pedestrian instances.

For every instance, we extract:

- Bounding box.
- Class label.
- Binary instance mask.

Person class label in Torchvision: 1.

### Dataset Split

- Training: 136 images (80%).
- Testing: 34 images (20%).
- Random seed: 42.

### DataLoader

Used to load data in batches.

Custom `collate_fn` is necessary because images can have different dimensions and contain different numbers of objects.

## 9. Object Detection Evaluation

### Confusion Matrix Terms

- True Positive (TP): Correct object detection.
- False Positive (FP): Incorrect detection.
- False Negative (FN): Missed object.

### Precision

`Precision = TP / (TP + FP)`

Measures how many predicted detections are correct.

### Recall

`Recall = TP / (TP + FN)`

Measures how many actual objects were detected.

### Average Precision (AP)

AP summarizes precision across different recall levels using the precision-recall curve.

### Mean Average Precision (mAP)

Mean Average Precision averages AP across evaluated classes.

Important variants:

- `mAP@0.50`: AP at IoU threshold 0.50.
- `mAP@0.75`: AP at IoU threshold 0.75.
- `mAP@0.50:0.95`: Average AP across IoU thresholds from 0.50 to 0.95.

For segmentation evaluation, IoU is calculated using pixel masks rather than bounding boxes.

### Mean Average Recall (mAR)

Measures recall under specified limits on the number of predictions per image.

Examples:
- `mAR@1`
- `mAR@10`
- `mAR@100`

## 10. Practical Faster R-CNN Results

Dataset: Penn-Fudan pedestrian dataset.

Training images: 136.

Testing images: 34.

Previously obtained results:

- mAP: 0.8000
- mAP@0.50: 0.9935
- mAP@0.75: 0.8954
- mAR@100: 0.8438

These are results from a small test split, not a guarantee of production performance.

## 11. Practical Mask R-CNN Results

Training: 5 epochs.

Training loss:

| Epoch | Loss |
|---|---:|
| 1 | 0.6020 |
| 2 | 0.2722 |
| 3 | 0.2169 |
| 4 | 0.1858 |
| 5 | 0.1710 |

The decreasing loss indicates that the model learned from the training data.

### Segmentation Evaluation

| Metric | Result |
|---|---:|
| Mask mAP | 0.7441 |
| Mask mAP@0.50 | 0.9943 |
| Mask mAP@0.75 | 0.9417 |
| Mask mAP small | 0.6139 |
| Mask mAP medium | 0.7318 |
| Mask mAP large | 0.7613 |
| Mask mAR@1 | 0.3767 |
| Mask mAR@10 | 0.7808 |
| Mask mAR@100 | 0.7808 |

The model performed better on medium and large instances than on small instances.

The Faster R-CNN and Mask R-CNN mAP values should not be compared directly because one evaluates bounding boxes and the other evaluates segmentation masks.

## 12. Important PyTorch Concepts

### Training Mode

`model.train()`

Activates training behavior.

### Evaluation Mode

`model.eval()`

Activates evaluation behavior.

### Disable Gradient Calculation

`torch.no_grad()`

Prevents gradient tracking during inference and evaluation.

### Backpropagation

`loss.backward()`

Calculates gradients for trainable parameters.

### Optimizer Update

`optimizer.step()`

Updates model parameters using calculated gradients.

### Clear Gradients

`optimizer.zero_grad()`

Clears accumulated gradients before the next training batch.

### Mask Probability to Binary Mask

Mask R-CNN predicts mask probabilities.

Applying a threshold converts them into binary masks:

`binary_mask = predicted_mask > 0.5`

## 13. Key Differences

| Feature | Faster R-CNN | Mask R-CNN |
|---|---|---|
| Classification | Yes | Yes |
| Bounding boxes | Yes | Yes |
| Pixel masks | No | Yes |
| Architecture | Two-stage | Two-stage with mask branch |
| Main task | Object detection | Instance segmentation |

## 14. What I Learned

- Difference between classification, detection and segmentation.
- Semantic segmentation vs instance segmentation.
- Bounding-box coordinates and IoU.
- NMS and its purpose.
- Faster R-CNN architecture and training.
- Mask R-CNN architecture and training.
- Dataset preparation using instance masks.
- Training and inference with Torchvision.
- Converting mask probabilities into binary masks.
- Visualizing individual instance masks.
- Evaluating detection and segmentation using mAP and mAR.

**Section completed:** Object Detection & Segmentation.