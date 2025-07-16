# CIFAR-100 Object Classification using CNN
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Download the Dataset
print("Loading CIFAR-100 dataset...")
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Reshaping the images - Prepare your dataset
print("\nPreparing dataset...")
# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(f"Data range after normalization: [{x_train.min()}, {x_train.max()}]")

# Encode the target labels
print("\nEncoding target labels...")
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

print(f"Encoded training labels shape: {y_train.shape}")
print(f"Encoded test labels shape: {y_test.shape}")

# Build the CNN with the specified architecture
print("\nBuilding CNN model...")
model = keras.Sequential([
    # 1. Input layer: 2D convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    
    # 2. Second 2D convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    
    # 3. 2D max pooling layer
    layers.MaxPooling2D((2, 2)),
    
    # 4. Dropout layer with 0.25 rate
    layers.Dropout(0.25),
    
    # 5. Two 2D convolutional layers with 64 units each
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 6. 2D max pooling layer
    layers.MaxPooling2D((2, 2)),
    
    # 7. Dropout layer with 0.25 rate
    layers.Dropout(0.25),
    
    # 8. Flattening layer
    layers.Flatten(),
    
    # 9. Fully connected layer with 512 units
    layers.Dense(512, activation='relu'),
    
    # 10. Dropout layer with 0.5 rate
    layers.Dropout(0.5),
    
    # 11. Output layer: fully connected with 100 units and softmax
    layers.Dense(100, activation='softmax')
])

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Choosing Hyper-parameters
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training Network
print("\nTraining the model...")
print("This may take a considerable amount of time (100 epochs)...")

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
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
actual_classes = np.argmax(y_test[:5], axis=1)

print("Sample predictions:")
for i in range(5):
    print(f"Image {i+1}: Predicted = {predicted_classes[i]}, Actual = {actual_classes[i]}")

# Display some test images with predictions
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i])
    plt.title(f'Predicted: {predicted_classes[i]}\nActual: {actual_classes[i]}')
    plt.axis('off')
plt.suptitle('Sample Test Images with Predictions')
plt.show()

# Calculate and display additional metrics
print(f"\nModel Performance Summary:")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

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

# Display final summary
print("\n" + "="*50)
print("CIFAR-100 CNN TRAINING COMPLETE")
print("="*50)
print(f"Dataset: CIFAR-100 (100 classes)")
print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image dimensions: {x_train.shape[1]}x{x_train.shape[2]}x{x_train.shape[3]}")
print(f"Training epochs: 100")
print(f"Batch size: 32")
print(f"Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("="*50)