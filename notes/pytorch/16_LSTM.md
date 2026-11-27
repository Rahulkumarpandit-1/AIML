# LSTM (Long Short-Term Memory)

## 1. What is LSTM?

LSTM stands for **Long Short-Term Memory**. It is a type of Recurrent Neural Network (RNN) designed to learn from sequential data while handling long-term dependencies more effectively than a basic RNN.

A basic RNN updates its hidden state at every time step. During training, information from earlier time steps can become difficult to preserve because of the vanishing gradient problem.

LSTM addresses this problem using a memory mechanism controlled by gates.

Examples of applications:
- Sentiment analysis
- Language modelling
- Speech recognition
- Time-series forecasting
- Sequential sensor data analysis

## 2. Why do we need LSTM?

Consider this sentence:

"The movie was not interesting, even though the actors performed well."

To understand the sentiment correctly, a model must consider the word "not" when interpreting "interesting".

A basic RNN may struggle to preserve useful information across a long sequence.

LSTM uses a cell state to carry information through the sequence and gates to control what information should be retained, added, or exposed.

**Important:** LSTM helps learn long-term dependencies, but it does not guarantee that every long-range relationship will be learned perfectly.

## 3. The two states in LSTM

LSTM maintains two important states.

### A. Cell State (Cₜ)

The cell state acts as the model's memory pathway.

- Carries information across time steps.
- Can preserve useful information for longer periods.
- Is controlled by the forget and input gates.

Think of it as the information the model chooses to keep in its memory.

### B. Hidden State (hₜ)

The hidden state represents the information exposed by the LSTM at the current time step.

- Contains the current output representation.
- Is passed to the next time step.
- Is used when producing predictions.

Think of it as the information the model currently exposes from its memory.

**Difference:**

| Cell State | Hidden State |
|---|---|
| Memory pathway | Current representation |
| Carries information forward | Exposes information for output |
| Updated through forget and input operations | Controlled by the output gate |

## 4. LSTM Gates

LSTM uses three gates:

1. Forget gate
2. Input gate
3. Output gate

A gate uses the sigmoid activation function, which produces values between 0 and 1.

- 0 means mostly blocked.
- 0.5 means partially allowed.
- 1 means mostly allowed.

These values are learned by the model. They are not manually assigned rules.

## 5. Forget Gate

The forget gate decides how much of the previous cell state should be retained.

Formula:

fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

Where:
- fₜ = forget gate output
- hₜ₋₁ = previous hidden state
- xₜ = current input
- Wf = learnable weights
- bf = learnable bias
- σ = sigmoid activation

The forget gate controls the previous cell state:

fₜ × Cₜ₋₁

Example:

Suppose:
- Previous cell state = 0.9
- Forget gate output = 0.2

Then:

0.2 × 0.9 = 0.18

Only 0.18 of the previous value remains in this simplified example.

If the forget gate output were 1, the previous cell-state value would be retained fully. If it were 0, that value would be removed.

**Important:** The forget gate acts on the previous cell state, not directly on the previous hidden state.

## 6. Input Gate

The input gate decides how much new candidate information should be added to the cell state.

Formula:

iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)

Where:
- iₜ = input gate output
- Wi = learnable weights
- bi = learnable bias

The input gate controls how much of the candidate information is written into memory.

A high input-gate value allows more candidate information into the cell state. A low value allows less.

## 7. Candidate Cell State

The candidate cell state proposes new information that could be stored.

Formula:

C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

Where:
- C̃ₜ = candidate cell state
- tanh = hyperbolic tangent activation
- Wc and bc = learnable parameters

The candidate is not itself a gate.

It proposes new information. The input gate controls how much of that information is accepted.

The tanh function produces values between -1 and 1.

## 8. Updating the Cell State

The cell state combines retained old information with accepted new information.

Formula:

Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ

There are two parts:

**Part 1: Retained information**

fₜ × Cₜ₋₁

The forget gate controls how much previous memory remains.

**Part 2: New information**

iₜ × C̃ₜ

The input gate controls how much candidate information is added.

### Numerical example

Suppose:

- Previous cell state = 0.9
- Forget gate = 0.2
- Input gate = 0.5
- Candidate cell state = 0.6

Retained information:

0.2 × 0.9 = 0.18

New information:

0.5 × 0.6 = 0.30

Updated cell state:

Cₜ = 0.18 + 0.30 = 0.48

The updated cell state is 0.48.

This is a simplified numerical example. In a real LSTM, these operations are performed on vectors.

## 9. Output Gate

The output gate decides how much information from the updated cell state should be exposed through the hidden state.

Formula:

oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)

The output gate produces values between 0 and 1.

It controls which parts of the updated memory are exposed.

## 10. Updating the Hidden State

The hidden state is calculated using the output gate and updated cell state.

Formula:

hₜ = oₜ × tanh(Cₜ)

Steps:

1. Apply tanh to the updated cell state.
2. Multiply the result by the output gate.
3. The result becomes the new hidden state.

The new hidden state is used as the current representation and passed to the next time step.

**Important:** The output gate controls the new hidden state. It does not directly decide how much previous cell-state information is retained.

## 11. Complete LSTM Workflow

At each time step, LSTM performs these operations:

1. Receives the current input xₜ.
2. Receives the previous hidden state hₜ₋₁.
3. Receives the previous cell state Cₜ₋₁.
4. The forget gate decides what previous memory to retain.
5. The input gate decides how much candidate information to add.
6. The candidate cell state proposes new information.
7. The cell state is updated.
8. The output gate controls what information is exposed.
9. The hidden state is updated.
10. The new hidden state and cell state are passed to the next time step.

The process repeats for every time step in the sequence.

## 12. LSTM Architecture

An LSTM receives three things at each time step:

- Current input: xₜ
- Previous hidden state: hₜ₋₁
- Previous cell state: Cₜ₋₁

It produces:

- New hidden state: hₜ
- New cell state: Cₜ

The hidden and cell states carry information to the next time step.

## 13. LSTM in PyTorch

PyTorch provides LSTM through:

`nn.LSTM`

Example:

```python
import torch
from torch import nn

lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    batch_first=True
)

x = torch.randn(64, 8, 10)

output, (hidden, cell) = lstm(x)

print(output.shape)
print(hidden.shape)
print(cell.shape)
```

### Understanding the parameters

**input_size=10**

Each time step contains 10 input features.

**hidden_size=20**

The hidden state has 20 features.

**batch_first=True**

The input shape follows:

(batch_size, sequence_length, input_size)

### Understanding the input

Input shape:

(64, 8, 10)

This means:
- 64 sequences in the batch
- 8 time steps per sequence
- 10 features at each time step

### Understanding the output

For a one-layer, unidirectional LSTM:

`output.shape` = (64, 8, 20)

It contains the output representation for every time step.

`hidden.shape` = (1, 64, 20)

It contains the final hidden state.

`cell.shape` = (1, 64, 20)

It contains the final cell state.

The first dimension represents the number of layers multiplied by the number of directions.

## 14. Using LSTM for Classification

For a task such as sentiment classification:

1. Convert words into token IDs.
2. Convert token IDs into embeddings.
3. Pass embeddings through the LSTM.
4. Take the final hidden state.
5. Pass it through a linear layer.
6. Produce class logits.

Example:

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        final_hidden = hidden[-1]

        logits = self.fc(final_hidden)

        return logits
```

### Understanding the data flow

**1. Embedding**

`embedded = self.embedding(x)`

Converts token IDs into dense numerical vectors.

**2. LSTM**

`output, (hidden, cell) = self.lstm(embedded)`

Processes the sequence and returns:
- Output at every time step
- Final hidden state
- Final cell state

**3. Final hidden state**

`final_hidden = hidden[-1]`

Selects the final hidden state from the last LSTM layer.

For a one-layer, unidirectional LSTM, its shape is:

(batch_size, hidden_size)

**4. Classification layer**

`logits = self.fc(final_hidden)`

Converts the final sequence representation into class scores.

For binary sentiment classification, the two logits represent the two classes.

## 15. LSTM vs Basic RNN

| Basic RNN | LSTM |
|---|---|
| Maintains hidden state | Maintains hidden and cell states |
| Simpler architecture | More complex architecture |
| No dedicated gates | Uses three gates |
| Can struggle with long-term dependencies | Designed to handle long-term dependencies better |
| Fewer parameters | More parameters |

LSTM is useful when information from earlier parts of a sequence may be important later.

## 16. LSTM vs GRU

| LSTM | GRU |
|---|---|
| Three gates | Two gates |
| Separate cell and hidden states | Only hidden state |
| More complex | Simpler |
| Often has more parameters | Often has fewer parameters |
| Can be useful for complex dependencies | Can be useful when a simpler recurrent model is preferred |

Neither model is always better. The result depends on the dataset, task, model configuration, and training process.

## 17. Advantages of LSTM

- Better at learning long-term dependencies than a basic RNN in many tasks.
- Uses gates to control information flow.
- Maintains a separate cell-state pathway.
- Can be applied to text, time-series, and other sequential data.

## 18. Limitations of LSTM

- More computationally expensive than a basic RNN.
- More complex because it has several gates and states.
- Processes sequence steps recurrently, which limits parallel processing across time steps.
- Can still struggle with very long sequences.
- Requires suitable training data and tuning.

## 19. Key Points to Remember

- LSTM stands for Long Short-Term Memory.
- It is a gated form of RNN.
- It maintains a cell state and a hidden state.
- Forget gate controls retained previous memory.
- Input gate controls how much candidate information is added.
- Candidate cell state proposes new information.
- Output gate controls what information is exposed.
- Cell state carries memory through the sequence.
- Hidden state represents the information exposed at the current time step.
- PyTorch uses `nn.LSTM`.

## 20. Final Summary

LSTM improves upon a basic RNN by introducing a cell state and three gates that control information flow.

The forget gate controls old memory, the input gate controls new information, and the output gate controls the information exposed through the hidden state.

In PyTorch, `nn.LSTM` provides this architecture and can be used for classification, language tasks, and time-series problems.
