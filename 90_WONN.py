# Group Number : 90
# Roll Numbers : Debraj Das (21ME30078)
# Project Number : WONN
# Project Title : Wine Origin Prediction using Artificial Neural Networks

# Importing the pytorch libraries for compare My model with Pytorch model
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset from a wine.data file
file_path = "./wine.data"
columns = ['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
           'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', '0D280_0D315_of_diluted_wines', 'Proline']

df = pd.read_csv(file_path, names=columns)

features = df.drop('class', axis=1)
target = df['class']

scaler = StandardScaler()
scaler.fit(features)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

y_train = y_train.values
y_test = y_test.values

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def forward(X, parameters, activation_function):
    Z1 = np.dot(X, parameters['W1']) + parameters['b1']
    A1 = activation_function(Z1)
    Z2 = np.dot(A1, parameters['W2']) + parameters['b2']
    A2 = softmax(Z2)
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}


def one_hot_encode_labels(y, num_classes):
    one_hot_labels = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        one_hot_labels[i, int(y[i]) - 1] = 1
    return one_hot_labels


def backward(X, y, forward_cache, parameters, learning_rate):
    m = X.shape[0]

    dZ2 = forward_cache['A2'] - \
        one_hot_encode_labels(y, forward_cache['A2'].shape[1])
    dW2 = (1/m) * np.dot(forward_cache['A1'].T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, parameters['W2'].T) * (forward_cache['A1'] > 0)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    parameters['W1'] -= learning_rate * dW1
    parameters['b1'] -= learning_rate * db1
    parameters['W2'] -= learning_rate * dW2
    parameters['b2'] -= learning_rate * db2


def train_model(X, y, parameters, activation_function, learning_rate, epochs, batch_size):
    m = X.shape[0]

    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            x_batch = X[i:i+batch_size, :]
            y_batch = y[i:i+batch_size]

            forward_cache = forward(x_batch, parameters, activation_function)
            backward(x_batch, y_batch, forward_cache,
                     parameters, learning_rate)


def evaluate(X, y, parameters, activation_function):
    forward_cache = forward(X, parameters, activation_function)

    predictions = np.argmax(forward_cache['A2'], axis=1)+1
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions, zero_division=0)

    m = X.shape[0]
    log_probs = -np.log(forward_cache['A2'][range(m), y-1])
    loss = np.sum(log_probs)/m

    return accuracy, loss, report


learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
architectures = [(X_train.shape[1], 0, 3),
                 (X_train.shape[1], 32, 3), (X_train.shape[1], 64, 3)]
activation_functions = [relu, relu, relu]
output_activation = softmax

results = []
best_model = [0, 0, 0, 0]
best_report = None

for arch, activation_function in zip(architectures, activation_functions):
    input_size, hidden_size, output_size = arch
    accuracies = []

    for lr in learning_rates:
        parameters = initialize_parameters(
            input_size, hidden_size, output_size)
        train_model(X_train, y_train, parameters,
                    activation_function, lr, epochs=100, batch_size=32)
        accuracy, loss, report = evaluate(
            X_test, y_test, parameters, activation_function)
        accuracies.append([hidden_size, lr, accuracy, loss])
        if accuracy > best_model[2]:
            best_model = [hidden_size, lr, accuracy, loss]
            best_report = report

    results.append(accuracies)

df = pd.DataFrame([item for sublist in results for item in sublist], columns=[
                  'Hidden Size', 'Learning Rate', 'Accuracy', 'Loss'])
print(df)

print("\nBest Model: ")
print("Hidden Size: ", best_model[0])
print("Learning Rate: ", best_model[1])
print("Accuracy: ", best_model[2])
print("Loss: ", best_model[3])

print("Best Model Classification Report: ", best_report)


def plot_graph(x, y, xname, yname, hidden):
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title('Plot of ' + yname + ' vs ' + xname +
              ' , Hidden Layer ' + str(hidden))
    plt.grid(True)
    plt.show()


for result in results:
    accuracies = []
    losses = []
    for point in result:
        accuracies.append(point[2])
        losses.append(point[3])
    plot_graph(learning_rates, accuracies,
               "Learning Rates", "Accuracy", result[0][0])
    plot_graph(learning_rates, losses, "Learning Rates", "Loss", result[0][0])


# pytorch library use to model the same dataset


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
        loss = criterion(outputs, y_test).item()
        return accuracy, loss


Hidden_sizes = [0, 32, 64]
Learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

results = []
best_accuracy = [0, 0, 0, 0]

for hidden_size in Hidden_sizes:
    for learning_rate in Learning_rates:
        accuracy, loss = models(hidden_size, learning_rate)
        results.append([hidden_size, learning_rate, accuracy, loss])
        if accuracy > best_accuracy[2]:
            best_accuracy = [hidden_size, learning_rate, accuracy, loss]

df = pd.DataFrame(results, columns=[
                  'Hidden Size', 'Learning Rate', 'Accuracy', 'Loss'])
print("Dataframe of the results using pytorch library:")
print(df)

print('Best model use pytorch library:')
print('Hidden Size:', best_accuracy[0])
print('Learning Rate:', best_accuracy[1])
print('Accuracy:', best_accuracy[2])
print('Loss:', best_accuracy[3])
