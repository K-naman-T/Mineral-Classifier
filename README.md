# Mineral Classifier

The Mineral Classifier is a Python program that uses Convolutional Neural Networks (CNNs) to classify images of minerals into different classes. It utilizes the PyTorch library for deep learning and image processing tasks.

## Features

- Loads and preprocesses a dataset of mineral images.
- Trains a CNN model to classify minerals into different classes.
- Evaluates the trained model on a test set and provides accuracy metrics.
- Generates a classification report with detailed performance metrics for each mineral class.

## Requirements

To run the Mineral Classifier, you need to have the following dependencies installed:

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- scikit-learn

## Usage

1. Clone the repository or download the `mineral_classifier.py` file.

2. Install the required dependencies using pip:

   ```shell
   pip install torch opencv-python numpy scikit-learn

    Set the appropriate dataset directory in the code:

    dataset_dir = '<path_to_dataset_directory>'

Run the program:

      python mineral_classifier.py

    The program will load and preprocess the dataset, train the CNN model, and evaluate its performance. The accuracy and classification report will be displayed in the console.

Dataset

The dataset used for training and testing the Mineral Classifier should follow the following structure:

dataset_directory/
├── mineral_class_1/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── ...
├── mineral_class_2/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── ...
├── ...

Each mineral class should have its own subdirectory within the dataset directory, and the images of each class should be stored within the respective subdirectory.
License

This project is licensed under the MIT License.

Feel free to modify and use the code according to your needs.
Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
Authors

    Kumar Naman Tiwary

Acknowledgments

    This project is inspired by the need for accurate mineral classification in the field of mining and geology.
    The code structure and implementation are based on best practices and references to PyTorch and computer vision documentation.

Please make sure to update the placeholders, such as `<path_t
