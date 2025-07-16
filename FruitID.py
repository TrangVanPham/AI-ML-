# Fruit Identification Mini Project

import os
import numpy as np
import matplotlib as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# --- Set file paths ---

DATASET_DIR = 'PATH_TO_FRUITS_360/fruits-360'
TRAIN_DIR = os.path.join(DATASET_DIR, 'Training')
TEST_DIR = os.path.join(DATASET_DIR, 'Test')

# --- Data Preparation ---
# Target image size required by dataset/instructions
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

# Define ImageDataGenerators for data augmentation and scaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

# 70% train, 30% validation (as test)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Map class indices to labels for later use
class_indices = train_generator.class_indices
labels = list(class_indices.keys())

# --- CNN Model Definition ---
model = Sequential([
    # 1st Conv layer
    Conv2D(16, (2, 2), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    # 2nd Conv layer
    Conv2D(32, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    # 3rd Conv layer
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    # 4th Conv layer
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(150, activation='relu'),
    Dropout(0.4),
    Dense(81, activation='softmax')  # Change to number of classes you have (here: 81 as per project, adjust if needed)
])

# --- Compilation ---
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Model Summary ---
print("\n--- Model Summary ---")
model.summary()

# --- Model Training ---
epochs = 30
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# --- Evaluation on Validation (Test) Data ---
validation_generator.reset()  # ensure generator starts from beginning
preds = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = validation_generator.classes

accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest accuracy: {accuracy:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap='Blues', annot=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
