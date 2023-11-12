import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import SST
from torchtext.data import Field, BucketIterator
from torchtext.vocab import GloVe

# Define Fields
TEXT = Field(lower=True, fix_length=200, batch_first=True)
LABEL = Field(sequential=False)

# Load SST dataset
train, valid, test = SST.splits(TEXT, LABEL)

# Build vocabulary and load GloVe vectors
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100), max_size=20000, min_freq=10)
LABEL.build_vocab(train)

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, nhead=2, dim_feedforward=hidden_dim), num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded)
        pooled = self.pooling(encoded.permute(0, 2, 1))
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)
        return output

# Create model, optimizer, and loss function
model = TextClassificationModel(len(TEXT.vocab), 100, 128, 2, len(LABEL.vocab))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Create data iterators
batch_size = 16
train_iter, valid_iter, test_iter = BucketIterator.splits((train, valid, test), batch_size=batch_size, repeat=False)

# Training loop
def train_model(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# Validation loop
def evaluate_model(model, iterator, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            total_loss += loss.item()
            predicted_labels = predictions.argmax(1)
            correct += (predicted_labels == batch.label).sum().item()
    return total_loss / len(iterator), correct / len(iterator.dataset)

# Train the model
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_model(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate_model(model, valid_iter, criterion)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation Acc: {valid_acc*100:.2f}%')

# Test the model
test_loss, test_acc = evaluate_model(model, test_iter, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')