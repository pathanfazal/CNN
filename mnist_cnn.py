import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset and split it into training and testing sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input images
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Convert the labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a sequential model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer with 32 filters and 3x3 kernel, ReLU activation
    MaxPooling2D((2, 2)),  # Max pooling layer with 2x2 pool size
    Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer with 64 filters and 3x3 kernel, ReLU activation
    MaxPooling2D((2, 2)),  # Max pooling layer with 2x2 pool size
    Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer with 64 filters and 3x3 kernel, ReLU activation
    Flatten(),  # Flatten the output from previous layer
    Dense(64, activation='relu'),  # Dense (fully connected) layer with 64 units, ReLU activation
    Dense(10, activation='softmax')  # Dense (fully connected) layer with 10 units (for 10 classes), softmax activation
])

# Compile the model with an optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)

# Print the test loss and accuracy
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

import numpy as np
import matplotlib.pyplot as plt

# Choose the number of random images to select
num_images = 5

# Randomly select indices from the test set without replacement
random_indices = np.random.choice(X_test.shape[0], num_images, replace=False)

# Get the sample images and labels using the random indices
sample_images = X_test[random_indices]
sample_labels = y_test[random_indices]

# Predict the labels for the sample images using the trained model
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Display the sample images and their predicted labels
fig, axs = plt.subplots(1, num_images, figsize=(12, 4))
for i in range(num_images):
    axs[i].imshow(sample_images[i].reshape(28, 28), cmap='gray')  # Plot the image
    axs[i].axis('off')
    axs[i].set_title(f"Predicted: {predicted_labels[i]}\nTrue: {np.argmax(sample_labels[i])}")  # Set the title
plt.show()
