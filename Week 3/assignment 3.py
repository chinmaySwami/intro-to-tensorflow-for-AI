import tensorflow as tf


class my_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') >= 0.998:
            print("\n Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


callback = my_callback()

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Reshaping the dataset to fit the needs of the convolutional layer
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Normalizing the dataset
training_images = training_images / 255
test_images = test_images / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model fitting
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callback])

# Using models for predictions on test set
loss= model.evaluate(test_images, test_labels)
