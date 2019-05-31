# TensorFlow and tf.keras
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import os
import h5py
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# model.compile(optimizer=tf.train.AdamOptimizer(),
#
#      loss='sparse_categorical_crossentropy',
#      metrics=['accuracy'])

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('classification.h5')
new_model.summary()

# test_loss, test_acc = new_model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)
#
predictions = new_model.predict(test_images)
# # now model has predict the model for each image in the testing set
print(predictions[2])
#
print(np.argmax(predictions[2]))
plt.imshow(test_images[2])

