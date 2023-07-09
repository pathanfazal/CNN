# CNN
MNIST Digit Recognition using Convolutional Neural Networks (CNN)
# MNIST Digit Recognition using Convolutional Neural Networks (CNN)

This repository contains code that demonstrates the training and testing of a Convolutional Neural Network (CNN) model for MNIST digit recognition using TensorFlow and Keras.

## Prerequisites

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- Keras

## Installation
#pip install tensorflow keras matplotlib
## Usage
#python mnist_cnn.py
This will train the CNN model on the MNIST dataset and evaluate its performance on the testing data.

After training and evaluation, the script will display a few randomly selected images from the test set along with their predicted labels.The code uses the MNIST dataset, which contains images of handwritten digits and their corresponding labels

After training, the model is evaluated on the testing set using the evaluate function. This computes the loss and accuracy of the model on the unseen testing data. The loss represents how well the model is performing, while the accuracy indicates the percentage of correctly predicted labels.

To visualize the model's predictions, a few random images from the testing set are selected. The model is then used to predict the labels for these images.

The predicted labels and the corresponding true labels are displayed alongside the images. This allows us to see how well the model performed on these specific examples.

Finally, the code generates a plot showing the sample images, their predicted labels, and their true labels.
