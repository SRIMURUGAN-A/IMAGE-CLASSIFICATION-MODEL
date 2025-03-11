# IMAGE-CLASSIFICATION-MODEL
"COMPANY" : CODTECH IT SOLUTIONS 
"NAME" : SRIMURUGAN
"INTERN ID" : CT08VRH 
"DOMAIN" : MACHINE LEARNING 
"DURATION" : 4 WEEKS 
"MENTOR" : Muzammil Ahmed

This script implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model is trained to recognize 10 different categories of objects, such as airplanes, cars, and animals.

Key Steps in the Code:
1. Importing Required Libraries:
tensorflow.keras: To build and train the CNN model.
matplotlib.pyplot: To visualize the training accuracy.
tensorflow.keras.datasets.cifar10: To load the CIFAR-10 dataset.
tensorflow.keras.utils.to_categorical: To convert labels into a categorical format for multi-class classification.
2. Loading and Preprocessing the Dataset:
CIFAR-10 dataset is loaded using cifar10.load_data().
Normalization: The pixel values (0-255) are scaled to a range of 0 to 1 for better model performance.
One-hot encoding: Labels are converted into categorical format (e.g., [0,1,0,...] for class 1).
3. Building the CNN Model:
The model consists of three convolutional layers followed by max-pooling layers to extract important features from images.
Flatten layer converts the extracted features into a 1D vector.
Dense (Fully Connected) layers:
One hidden layer with 128 neurons (ReLU activation).
Output layer with 10 neurons (Softmax activation) for multi-class classification.
4. Compiling the Model:
Optimizer: adam is used to adjust weights efficiently.
Loss function: categorical_crossentropy is used since it’s a multi-class classification problem.
Evaluation metric: accuracy is used to measure performance.
5. Training the Model:
The model is trained for 10 epochs with a batch size of 64.
The validation set (X_test, y_test) is used to monitor performance during training.
6. Evaluating the Model:
The test accuracy is printed after training.
A training history plot shows how the accuracy changes over epochs for both training and validation datasets.
