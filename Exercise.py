# MNIST Handwritten Digit Classification using CNN
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Download the Dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Step 2: Prepare your dataset
print("\nPreparing dataset...")
# Reshape and normalize the data
x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
x_test = x_test.reshape(10000, 28, 28, 1) / 255.0

print(f"Reshaped training data: {x_train.shape}")
print(f"Reshaped test data: {x_test.shape}")
print(f"Data range: [{x_train.min()}, {x_train.max()}]")

# Define Your Convolutional Neural Network
print("\nBuilding CNN model...")
model = keras.Sequential([
    # Input layer: 2D convolutional layer
    layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # First hidden layer: 2D max pooling layer
    layers.MaxPooling2D((2, 2)),
    
    # Second hidden layer: flattening layer
    layers.Flatten(),
    
    # Third hidden layer: fully connected layer
    layers.Dense(128, activation='relu'),
    
    # Output layer: fully connected layer with softmax
    layers.Dense(10, activation='softmax')
])

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Choosing Hyperparameters
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training Network
print("\nTraining the model...")
print("This may take a few minutes...")

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_split=0.1,
    verbose=1
)

# Test the model
print("\nEvaluating model on test set...")
test_results = model.evaluate(x_test, y_test, verbose=0)
test_loss, test_accuracy = test_results

print(f"\nFinal Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions on a few test samples
print("\nMaking predictions on sample test images...")
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)

print("Sample predictions:")
for i in range(5):
    print(f"Image {i+1}: Predicted = {predicted_classes[i]}, Actual = {y_test[i]}")

# Display some test images with predictions
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_classes[i]}\nActual: {y_test[i]}')
    plt.axis('off')
plt.show()

# Return the model and test results as requested
print(f"\nReturning model and test results:")
print(f"Model: {model}")
print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# Function to return the required outputs
def get_model_and_results():
    """
    Returns the trained model and test results as specified in requirements
    """
    return model, (test_loss, test_accuracy)