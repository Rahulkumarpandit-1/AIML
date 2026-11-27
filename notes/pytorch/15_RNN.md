# 15. Recurrent Neural Networks (RNN)

## 1. What is an RNN?

A Recurrent Neural Network (RNN) is a neural network designed to process sequential data.

Unlike a traditional feedforward neural network, an RNN maintains a hidden state that carries information from previous time steps.

This allows it to use earlier information when processing the current input.

### Examples of sequential data

- Text and sentences
- Speech and audio
- Time-series data
- Stock prices
- Sensor readings
- Video frames

**Core idea:** An RNN processes data step by step while maintaining information from previous steps.

## 2. Why Do We Need RNNs?

Traditional neural networks generally process inputs without maintaining information about previous inputs.

But in sequential data, order matters.

Example:

- "I love this movie."
- "I don't love this movie."

The word order and surrounding context affect the meaning.

An RNN processes words in sequence and carries information forward through its hidden state.

## 3. How Does an RNN Work?

An RNN processes one input at a time.

Example sentence:

"I am learning"

Processing:

1. Input "I" → produces hidden state h1.
2. Input "am" → uses h1 and produces h2.
3. Input "learning" → uses h2 and produces h3.

The hidden state carries information from earlier time steps.

### Hidden State

The hidden state is an internal numerical representation of information accumulated by the RNN.

It is updated at every time step.

## 4. RNN Architecture

An RNN uses the same recurrent cell and shared weights at every time step.

At each time step, it receives:

- Current input: x_t
- Previous hidden state: h_(t-1)

It produces:

- New hidden state: h_t
- Optional output: y_t

### Hidden State Equation

h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

Where:

- x_t = current input
- h_(t-1) = previous hidden state
- W_xh = weights applied to the current input
- W_hh = recurrent weights applied to the previous hidden state
- b_h = bias
- tanh = activation function

The tanh function introduces non-linearity and produces values between -1 and 1.

### Output Equation

y_t = W_hy h_t + b_y

Where:

- h_t = current hidden state
- W_hy = output weights
- b_y = output bias
- y_t = output at the current time step

The output can represent prediction scores.

## 5. Types of RNN Architectures

### One-to-One

One input produces one output.

Example: Traditional image classification.

### One-to-Many

One input produces a sequence of outputs.

Example: Image caption generation.

### Many-to-One

A sequence of inputs produces one output.

Example: Sentiment classification.

### Many-to-Many

A sequence of inputs produces a sequence of outputs.

Examples:
- Machine translation
- Sequence labelling
- Part-of-speech tagging

## 6. Training an RNN

RNNs are trained using Backpropagation Through Time (BPTT).

During forward processing, the RNN reads the sequence and produces predictions.

During backward processing, gradients are propagated through the time steps to update the shared parameters.

This allows the model to learn from sequential information.

## 7. Limitations of Traditional RNNs

### Vanishing Gradient Problem

During backpropagation through time, gradients can become extremely small as they move through many time steps.

As a result, earlier time steps receive weak learning signals.

This makes it difficult for a traditional RNN to learn long-term dependencies.

Example:

In a long sentence, an important word appearing near the beginning may have a relationship with a word near the end. A basic RNN may struggle to retain that information.

### Exploding Gradient Problem

Gradients can also become extremely large during backpropagation.

This can cause unstable training and very large parameter updates.

Gradient clipping is one technique used to control exploding gradients.

## 8. Long-Term Dependencies

A long-term dependency occurs when information from an earlier time step is important for understanding or predicting something much later in the sequence.

Example:

"The movie was released several years ago, and although many people initially disliked it, I still think it was excellent."

Understanding the overall sentiment may require connecting information across distant words.

Traditional RNNs can struggle with these long-range relationships.

## 9. RNN vs Feedforward Neural Network

| Feedforward Neural Network | RNN |
|---|---|
| Processes inputs through forward layers | Processes sequential inputs step by step |
| Does not maintain recurrent hidden state | Maintains hidden state |
| No inherent sequence memory | Carries information from previous steps |
| Commonly used for fixed-size inputs | Useful for sequential data |

## 10. Practical RNN Implementation in PyTorch

PyTorch provides `nn.RNN` for building recurrent neural networks.

### Create an RNN layer

```python
import torch
from torch import nn

rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    batch_first=True
)
```

Parameters:

- `input_size`: Number of features in each time-step input.
- `hidden_size`: Number of values in the hidden state.
- `batch_first=True`: Input shape follows (batch_size, sequence_length, features).

### Prepare sample input

```python
input_data = torch.randn(64, 8, 10)
```

Meaning:

- 64 = number of sequences in the batch.
- 8 = time steps per sequence.
- 10 = features per time step.

### Pass input through RNN

```python
output, hidden = rnn(input_data)

print(output.shape)
print(hidden.shape)
```

For a single-layer, unidirectional RNN:

```text
Output: torch.Size([64, 8, 20])
Hidden: torch.Size([1, 64, 20])
```

### Understanding the outputs

**Output:**

Contains the hidden representation for every time step.

Shape: (batch_size, sequence_length, hidden_size)

**Hidden:**

Contains the final hidden state for each recurrent layer and direction.

Shape: (num_layers, batch_size, hidden_size) for a single-direction RNN.

## 11. RNN Applications

- Sentiment analysis
- Text classification
- Speech recognition
- Time-series prediction
- Sequence labelling
- Language modelling

## 12. Key Takeaways

- RNNs are designed for sequential data.
- They process inputs step by step.
- Hidden states carry information from previous time steps.
- The same recurrent weights are shared across time steps.
- BPTT is used to train RNNs.
- Vanishing and exploding gradients are important limitations.
- LSTM and GRU introduce gates to improve information handling over long sequences.

**Core concept:** An RNN processes a sequence while maintaining a hidden state that carries information from previous time steps.