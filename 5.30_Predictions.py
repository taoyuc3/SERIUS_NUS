from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

# load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist

# convert the samples from integers to floating-point numbers
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras model by stacking layers.
# Select an optimizer and loss function used for training
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# now we are going to train and evaluate the model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)





