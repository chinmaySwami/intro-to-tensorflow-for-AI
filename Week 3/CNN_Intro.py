import tensorflow as tf
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from visualizing_cnn_layers import visualize_layer_results


# Download the MNIST dataset
mnist = tf.keras.datasets.mnist

# Splitting the dataset into train and test
(training_images, training_labels),  (test_images, test_labels) = mnist.load_data()

# Reshaping the dataset to fit the needs of the convolutional layer
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Normalizing the dataset
training_images = training_images/255
test_images = test_images/255

# Defining the CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()  # prints the layer details of NN
model.fit(training_images, training_labels, epochs=5)

loss= model.evaluate(test_images, test_labels)

visualize_layer_results(model, test_images)