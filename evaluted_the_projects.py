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

print("Best Model Classification Report: ")
print(best_report)
