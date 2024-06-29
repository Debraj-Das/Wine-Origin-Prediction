import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('./wine.data')


features = df.iloc[:, 1:].values
labels = df.iloc[:, 0].values - 1  # Adjust labels to start from 0 instead of 1

X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_size = len(df.columns) - 1
output_size = 3
epochs = 100


def models(hidden_size, learning_rate):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
        accuracy = torch.sum(predictions == y_test).item() / len(y_test)
        report = classification_report(y_test, predictions, zero_division=0)
        loss = criterion(outputs, y_test).item()
        return accuracy, loss, report


Hidden_sizes = [0, 32, 64]
Learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

results = []
best_accuracy = [0, 0, 0, 0]
best_report = None

for hidden_size in Hidden_sizes:
    for learning_rate in Learning_rates:
        accuracy, loss, report = models(hidden_size, learning_rate)
        results.append([hidden_size, learning_rate, accuracy, loss])
        if accuracy > best_accuracy[2]:
            best_accuracy = [hidden_size, learning_rate, accuracy, loss]
            best_report = report

df = pd.DataFrame(results, columns=[
                  'Hidden Size', 'Learning Rate', 'Accuracy', 'Loss'])
print(df)

print('Best model:')
print('Hidden Size:', best_accuracy[0])
print('Learning Rate:', best_accuracy[1])
print('Accuracy:', best_accuracy[2])
print('Loss:', best_accuracy[3])

print("Classification Report:")
print(best_report)
