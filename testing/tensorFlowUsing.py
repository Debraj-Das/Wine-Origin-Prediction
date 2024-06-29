import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your dataset, assuming the file is in CSV format
# Replace 'your_dataset.csv' with the actual filename
df = pd.read_csv('./wine.data')

# Separate features and target variable
X = df.iloc[:, 1:].values  # Assuming the first column is the class column
y = df.iloc[:, 0].values

# One-hot encode the target variable
y_one_hot = pd.get_dummies(y).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu',
                          input_shape=(X_train.shape[1],)),
    # 3 classes for output layer
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model with Adam optimizer and learning rate of 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train_scaled, y_train, epochs=100,
          batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print classification report
print(accuracy_score(y_test_classes, y_pred_classes))
