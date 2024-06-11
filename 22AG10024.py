# NAME:- Mantej Singh Sohi
# Roll NO.:- 22AG10024

from google.colab import drive
drive.mount('/content/gdrive')
!unzip gdrive/MyDrive/lab_test_2_dataset.zip

# EXPERIMENT 1

import torch
import numpy as np
import random

# setting seed value
seed_value = 2022
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Set device to CPU
device = torch.device('cpu')

# EXPERIMENT 2

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load and preprocess data
data_path = '/content/lab_test_2_dataset'
image_size = (32, 32)
batch_size = 32

# Load images and labels
images = []
labels = []
for age_folder in sorted(os.listdir(data_path)):
    for image_file in os.listdir(os.path.join(data_path, age_folder)):
        image = load_img(os.path.join(data_path, age_folder, image_file), target_size=image_size)
        image = img_to_array(image) / 255.0
        images.append(image)
        labels.append(int(age_folder))

images = np.array(images)
labels = np.array(labels)

np.random.seed(2022)
shuffle_indices = np.random.permutation(len(images))
images_shuffled = images[shuffle_indices]
labels_shuffled = labels[shuffle_indices]

# Splitting dataset into train, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images_shuffled, labels_shuffled, test_size=0.15, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1765, random_state=42)  # 0.15 / (1 - 0.15)


# Printing dataset sizes
print(f"Overall Dataset Size: {len(images)}")
print(f"Training Dataset Size: {len(train_images)}")
print(f"Validation Dataset Size: {len(val_images)}")
print(f"Testing Dataset Size: {len(test_images)}")

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# EXPERIMENT 3

# Define the model
class CNN_Model(tf.keras.Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)  # Output layer for regression

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = CNN_Model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mse'])

# EXPERIMENT 4

epochs = 25
interval = 5  # Save model every 5 epochs
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0

    # Training
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = model.compiled_loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_epoch_loss += loss.numpy()

    # Validation
    for val_images, val_labels in val_dataset:
        val_predictions = model(val_images)
        val_loss = model.compiled_loss(val_labels, val_predictions)
        val_epoch_loss += val_loss.numpy()

    # Calculate average losses
    train_epoch_loss /= len(train_dataset)
    val_epoch_loss /= len(val_dataset)
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    # Save model at intervals
    if (epoch + 1) % interval == 0:
        model.save_weights(f"model_epoch_{epoch+1}.h5")

# Generate plot for training and validation losses
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Evaluate the model on the testing dataset
test_losses = []
predicted_labels = []

for images, labels in test_dataset:
    predictions = model(images)
    test_loss = model.compiled_loss(labels, predictions)
    test_losses.append(test_loss.numpy())
    predicted_labels.extend(predictions.numpy())

# Calculate overall MSE
test_mse = mse(test_labels, np.array(predicted_labels))

print(f"Mean Squared Error on Testing Dataset: {test_mse:.4f}")

# Create scatter plot of predicted labels vs. ground truth labels
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, predicted_labels, color='blue', alpha=0.5)
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], color='red', linestyle='--')
plt.title('Predicted vs. Ground Truth Age Labels')
plt.xlabel('Ground Truth Age')
plt.ylabel('Predicted Age')
plt.grid(True)
plt.show()