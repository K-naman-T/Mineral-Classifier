## Overview:

This script builds a convolutional neural network (CNN) model for mineral image classification using the PyTorch library. It assumes a dataset structure where each mineral class corresponds to a subdirectory within a parent directory. Each subdirectory should contain images of the respective mineral class. 

The program performs the following key steps:

i) Loads and preprocesses the image data.
ii) Splits the data into training and testing sets.
iii) Defines a CNN model and a custom PyTorch Dataset.
iv) Trains the model using the training set.
v) Evaluates the model's performance on the testing set.

## Dependencies:

The following libraries are required to run this program:

- **os**: A module in Python that provides functions for interacting with the operating system.
- **cv2**: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library with bindings for Python.
- **numpy**: A library in Python that provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- **torch**: PyTorch is an open-source machine learning library for Python, based on Torch, used for applications such as natural language processing and artificial neural networks.
- **torch.nn**: A sub-library of PyTorch, it provides classes to create and manage neural network layers and defines a set of functions to compute common loss functions.
- **torch.optim**: A sub-library of PyTorch that contains standard optimization operations like SGD and Adam.
- **torch.utils.data**: A sub-library of PyTorch that provides utilities for loading and handling datasets in PyTorch.
- **sklearn.model_selection**: A module in scikit-learn that provides utilities for model selection, including methods to split datasets, generate validation curves, and tune parameters.
- **sklearn.metrics**: A module in scikit-learn that includes score functions, performance metrics, and pairwise metrics and distance computations.


## Detailed Description:

1. **Dataset Loading and Preproccessing**

The dataset used for this task is a collection of mineral images, which are classified into 7 different categories:

Biotite

Bornite

Chrysocolla

Malachite

Muscovite

Pyrite

Quartz

Each category corresponds to a unique type of mineral, and the images associated with each category are stored in separate folders within the parent dataset directory. The dataset hierarchy thus forms a structured and easily navigable format for segregating the images according to their mineral type.

In the data preprocessing step, each image is read from its respective folder, resized to a uniform size (100x100 pixels in this case), and the pixel data is appended to a list. Simultaneously, the label corresponding to each image (determined from the folder name) is also stored in a separate list.

The images are loaded, resized to 100x100 pixels, and appended to the images list. The corresponding label is determined by the folder name and appended to the labels list. The images and labels are then converted to NumPy arrays.

2.**Splitting the Dataset**

The dataset is divided into training and testing sets using the 'train_test_split' function from scikit-learn. 20% of the data is reserved for testing the model's performance.

3.**Converting Data to PyTorch Tensors**

The image data arrays are converted to PyTorch tensors, and the channel dimension is moved to the second dimension (required for PyTorch's Conv2d layer). The label arrays are also converted to tensors.

4.**Creating Data Loaders**

PyTorch's DataLoader is used to divide the data into batches and to shuffle the data. Separate loaders are created for the training and testing sets.

5. **Model**

The model used is a Convolutional Neural Network (CNN) defined by the MineralClassifier class. The CNN is a popular deep learning model used for image recognition tasks due to its ability to capture spatial dependencies in an image through the application of relevant filters.

The architecture of the MineralClassifier consists of:

A convolution layer (self.conv1), which applies 16 filters of size 3x3 on the input image. Each filter extracts different kinds of features from the input image. Padding is used to preserve spatial dimensions.

A pooling layer (self.pool), which reduces the spatial dimensions of the input while preserving the most important information. This layer uses max pooling with a 2x2 window and stride 2.

Two fully connected layers (self.fc1 and self.fc2), which perform classification based on the features extracted by the convolution and pooling layers. The first fully connected layer has 128 units and the final layer has 7 units, corresponding to the 7 classes of minerals.

The ReLU (Rectified Linear Unit) activation function is used in this model, which introduces non-linearity into the model allowing it to learn complex patterns.

(i) ***Loss Function***

The CrossEntropyLoss function is used as the loss function in this model. This is a common choice for classification problems, and especially so for multi-class classification like this one. Cross-Entropy Loss measures the performance of a classification model whose output is a probability value between 0 and 1. It increases as the predicted probability diverges from the actual label, providing a strong gradient for correct classification.

(ii) *Optimization Technique*

The Adam (Adaptive Moment Estimation) optimizer is used for training the model. Adam is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing. It's known for its efficiency and effective handling of sparse gradients, as well as keeping an exponentially decaying average of past gradients.

6. **Training the Model**

The model is trained over a specified number of epochs. In each epoch, the model's gradients are reset, a forward pass is performed on the training data, the loss between the output and actual labels is computed, backpropagation is performed to compute the gradients, and the optimizer updates the model's weights.

7. **Evaluating the Model

After training, the model is evaluated on the test data. The model's '.eval()' method is called to set it to evaluation mode, which turns off features like dropout. Predictions are made on the test data, and the accuracy of the model is computed and printed.


Finally, a classification report is generated using scikit-learn's 'classification_report' function, providing detailed metrics about the model's performance.
