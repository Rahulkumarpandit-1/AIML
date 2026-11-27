
import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.gru(embedded)

        final_hidden = hidden[-1]

        logits = self.fc(final_hidden)

        return logits


vocab_size = 10
embedding_dim = 8
hidden_size = 16

model = GRUClassifier(
    vocab_size,
    embedding_dim,
    hidden_size
)
vocab = {
    "i": 1,
    "love": 2,
    "this": 3,
    "movie": 4,
    "hate": 5,
    "boring": 6,
    "is": 7,
    "good": 8,
    "bad": 9
}
X_train = torch.tensor([
    [1, 2, 3, 4],
    [3, 4, 7, 8],
    [1, 5, 3, 4],
    [3, 4, 7, 6]
])
y_train=torch.tensor([1,1,0,0])

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs=100
for epoch in range(epochs):
    optimizer.zero_grad()
    logits=model(X_train)
    loss=loss_fn(logits,y_train)
    loss.backward()
    optimizer.step()
     
    if (epoch+1)%10==0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
     
X_test = torch.tensor([
    [1, 2, 3, 8],
    [1, 5, 3, 9]
])

y_test = torch.tensor([1, 0])

model.eval()

with torch.no_grad():
    test_logits = model(X_test)
    test_predictions = torch.argmax(test_logits, dim=1)


print("Test predictions:", test_predictions)
print("Actual labels:", y_test)