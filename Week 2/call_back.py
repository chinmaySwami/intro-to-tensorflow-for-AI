# Importing tensor flow and other related libraries
import tensorflow as tf
import numpy as np


class my_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss')<0.2):
            print("Cancelling the training::")
            self.model.stop_training = True

# Retriving the mnist dataset
mnist = tf.keras.datasets.mnist

# Splitting the dataset into train and test
(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

# Normalizing the dataset
training_images = training_images/2
test_images = test_images/2
callbacks = my_callback()

# creating the NN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # to get 2D matrix of image and convert it to 1D
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # Hidden layer of 512 neurons
    tf.keras.layers.Dense(256, activation=tf.nn.relu),  # Hidden layer of 256 neurons
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Output layer of 10 neurons because we have 10 classes
])

# Compiling the model arch
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training the model
# Using callbacks to stop after certain amount of training loss is reached
model.fit(training_images, training_labels, epochs=5, verbose=2, callbacks=[callbacks])

# Testing the model
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0], test_labels[0])


