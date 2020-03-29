import tensorflow as tf
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Creating a function that would create and return a model
def create_model():
    # creating the NN architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),  # to get 2D matrix of image and convert it to 1D
        tf.keras.layers.Dense(512, activation=tf.nn.relu),  # Hidden layer of 512 neurons
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Output layer of 10 neurons because we have 10 classes
    ])

    # Compiling the model arch
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

# Download the MNIST dataset
mnist = tf.keras.datasets.mnist

# Splitting the dataset into train and test
(training_images, training_labels),  (test_images, test_labels) = mnist.load_data()

# Normalizing the dataset
training_images = training_images/2
test_images = test_images/2

# Creating a model
model = KerasClassifier(build_fn=create_model, verbose=0)

# Creating parameter dictionary for hypertuning
params = {'epochs': [5, 6, 8, 10]}

# Creating grid search object
grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
grid_result = grid.fit(training_images, training_labels)

# Testing the model
grid_result.predict(test_images)

classifications = model.predict(test_images)

print(classifications[0], test_labels[0])
