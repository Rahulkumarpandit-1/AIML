# 14. Vision Transformers (ViT)

## 1. What is Vision Transformer?

Vision Transformer (ViT) is a deep learning architecture that applies the Transformer architecture to image classification.

Traditional CNNs use convolutional layers to extract image features. ViT divides an image into smaller patches and processes them as a sequence of tokens using Transformer encoder blocks.

**Main idea:** Instead of processing an entire image using convolution operations, ViT treats image patches similarly to how NLP Transformers process word tokens.

## 2. Why Vision Transformers?

CNNs are effective at identifying local image patterns such as edges, textures and shapes.

However, understanding relationships between distant image regions can require multiple convolutional layers.

Transformers use self-attention to learn relationships between different image patches.

ViT can therefore learn both local and global relationships between image regions.

## 3. How Vision Transformer Works

The main processing pipeline:

1. Input image
2. Image patch division
3. Patch embedding
4. Positional embedding
5. Class token addition
6. Transformer encoder
7. Classification head
8. Final prediction

## 4. Image Patching

ViT divides an image into fixed-size patches.

Example:

- Input image: 224 × 224 pixels
- Patch size: 16 × 16 pixels
- Number of patches: 14 × 14 = 196

Each patch is converted into a flattened representation before being projected into an embedding vector.

For an RGB image, each patch contains:

16 × 16 × 3 = 768 values.

## 5. Patch Embedding

Patch embedding converts each flattened image patch into a fixed-dimensional vector.

For ViT-B/16:

- Number of patches: 196
- Embedding dimension: 768

Therefore, each image patch becomes a vector containing 768 values.

These vectors act as image tokens.

## 6. Positional Embedding

Transformers do not automatically understand the original position of each token.

ViT adds positional embeddings to patch embeddings so the model can learn where each patch came from in the original image.

This helps preserve spatial information.

## 7. Class Token (CLS Token)

ViT adds a special learnable token called the classification token.

For ViT-B/16:

- Image patches: 196
- CLS token: 1
- Total tokens: 197

The CLS token interacts with the other tokens through self-attention.

Its final representation is used for image classification.

## 8. Transformer Encoder

The sequence of image tokens is passed through Transformer encoder blocks.

Each encoder block contains:

### Multi-Head Self-Attention

Self-attention allows image patches to exchange information and learn relationships with other patches.

For example, a patch containing part of an animal can learn relationships with patches containing other parts of the animal.

Multi-head attention learns different relationships in parallel.

### Feed-Forward Network

The feed-forward network processes each token representation and learns additional feature transformations.

### Residual Connections and Layer Normalization

Residual connections help information and gradients flow through the network.

Layer normalization stabilizes the processing of token representations.

## 9. Classification Head

After the Transformer encoder, the final CLS token representation is passed to a classification head.

For example, if the model has 1,000 output classes, the classification head produces 1,000 scores.

The highest-scoring class becomes the predicted class.

## 10. ViT-B/16 Architecture

ViT-B/16 means:

- B = Base model
- 16 = Patch size of 16 × 16 pixels

Important architecture details:

| Component | Configuration |
|---|---|
| Input image | 224 × 224 |
| Patch size | 16 × 16 |
| Number of patches | 196 |
| Total tokens | 197 |
| Embedding dimension | 768 |
| Transformer encoder blocks | 12 |
| Attention heads | 12 |

## 11. CNN vs Vision Transformer

| CNN | Vision Transformer |
|---|---|
| Uses convolution operations | Uses self-attention |
| Extracts local features through filters | Learns relationships between image patches |
| Has convolution-based inductive biases | Has less built-in image-specific inductive bias |
| Processes image feature maps | Processes image tokens |
| Often works well with smaller datasets | Often benefits from large-scale pretraining |

Neither architecture is universally better. Performance depends on the dataset, task, architecture and training method.

## 12. Practical Implementation Using PyTorch

We can use a pretrained ViT model from Torchvision.

### Import libraries

```python
import torch
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
```

### Load pretrained weights

```python
weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(weights=weights)

model.eval()
```

The pretrained weights contain parameters learned from a large image dataset.

### Load and preprocess an image

```python
image = Image.open("test_image.jpg").convert("RGB")

transform = weights.transforms()

image_tensor = transform(image).unsqueeze(0)
```

- `Image.open()` loads the image.
- `convert("RGB")` ensures three colour channels.
- `weights.transforms()` applies the preprocessing expected by the pretrained model.
- `unsqueeze(0)` adds the batch dimension.

### Make predictions

```python
with torch.no_grad():
    outputs = model(image_tensor)
```

The model returns classification scores called logits.

### Convert logits into probabilities

```python
probabilities = torch.softmax(outputs, dim=1)
```

Softmax converts the scores into probabilities.

### Get the top five predictions

```python
top_probabilities, top_indices = torch.topk(
    probabilities[0],
    k=5
)

categories = weights.meta["categories"]

for probability, index in zip(top_probabilities, top_indices):
    print(
        f"{categories[index.item()]}: "
        f"{probability.item() * 100:.2f}%"
    )
```

This displays the five classes with the highest predicted probabilities.

## 13. Important PyTorch Concepts

- `vit_b_16()` creates the ViT-Base architecture with 16 × 16 patches.
- `ViT_B_16_Weights.DEFAULT` loads the default pretrained weights.
- `model.eval()` switches the model into evaluation mode.
- `torch.no_grad()` disables gradient tracking during inference.
- `torch.softmax()` converts logits into probabilities.
- `torch.topk()` selects the highest-scoring predictions.
- `weights.meta["categories"]` provides the class names.

## 14. Limitations of Vision Transformers

1. Large models can require substantial computational resources.
2. Training from scratch can require large datasets.
3. Self-attention can be computationally expensive for long token sequences.
4. Pretrained models may not perform well on every specialised image domain without adaptation.

## 15. Key Takeaways

- ViT applies Transformer architecture to computer vision.
- Images are divided into patches.
- Patches are converted into embedding vectors.
- Positional embeddings provide spatial-position information.
- A CLS token represents the image for classification.
- Transformer encoders use self-attention to learn relationships between patches.
- The classification head produces final image-classification scores.
- ViT can be used through pretrained Torchvision models.

**Core concept:** ViT treats an image as a sequence of patches and uses Transformer self-attention to learn relationships between them.