# CNN
MNIST Digit Recognition using Convolutional Neural Networks (CNN)
# MNIST Digit Recognition using Convolutional Neural Networks (CNN)

This repository contains code that demonstrates the training and testing of a Convolutional Neural Network (CNN) model for MNIST digit recognition using TensorFlow and Keras.

## Prerequisites

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- Keras

## Installation
pip install tensorflow keras matplotlib
## Usage
python mnist_cnn.py
•	This will train the CNN model on the MNIST dataset and evaluate its performance on the testing data.
•	After training and evaluation, the script will display a few randomly selected images from the test set along with their predicted labels.The code uses the MNIST dataset, which contains images of handwritten digits and their corresponding labels
## Explanation
•	The dataset is divided into training and testing sets. The training set is used to train the neural network model, while the testing set is used to evaluate its performance.
•	The images in the dataset are grayscale and have a size of 28x28 pixels. Before training, the images are reshaped to a suitable format and normalized by dividing the pixel values by 255.0 to bring them within the range of 0 to 1.
•	The labels in the dataset are converted into a categorical format, where each label is represented as a one-hot encoded vector. This step is necessary for training the neural network.
•	A convolutional neural network (CNN) model is created using the Keras library. The model consists of several layers, including convolutional layers, pooling layers, and dense (fully connected) layers. These layers are designed to extract features from the input images and make predictions.
•	The model is compiled with an optimizer, loss function, and metrics. The optimizer (Adam) is responsible for adjusting the weights of the neural network during training to minimize the loss. The loss function (categorical cross-entropy) measures the difference between the predicted and actual labels. The accuracy metric is used to evaluate the model's performance.
•	The model is trained on the training set using the fit function. The training is performed in batches, with each batch containing a subset of the training data. The number of epochs determines how many times the entire training set is passed through the network. During training, the model adjusts its weights to improve its predictions.
•	After training, the model is evaluated on the testing set using the evaluate function. This computes the loss and accuracy of the model on the unseen testing data. The loss represents how well the model is performing, while the accuracy indicates the percentage of correctly predicted labels.
•	To visualize the model's predictions, a few random images from the testing set are selected. The model is then used to predict the labels for these images.
•	The predicted labels and the corresponding true labels are displayed alongside the images. This allows us to see how well the model performed on these specific examples.
•	Finally, the code generates a plot showing the sample images, their predicted labels, and their true labels.
