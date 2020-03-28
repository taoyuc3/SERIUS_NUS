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

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# test_loss, test_acc = new_model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)

predictions = new_model.predict(test_images)
# # now model has predict the model for each image in the testing set
print(predictions[100])
print("The prediction is:", np.argmax(predictions[100]), "-", class_names[np.argmax(predictions[100])])
print("The actual label is:", test_labels[100], "-", class_names[test_labels[100]])


# # now can graph this to look at the full set of 10 channels
# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
#
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#     # Let's look at the 0th image, predictions, and prediction array.
#     i = 0
#     plt.figure(figsize=(6, 3))
#     plt.subplot(1, 2, 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(1, 2, 2)
#     plot_value_array(i, predictions, test_labels)
#     plt.show()
#
#
# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)
# plt.show()

# Grab an image from the test dataset
img = test_images[0]

print(img.shape)
#
# # Now predict the image:
# predictions_single = new_model.predict(img)
#
# print(predictions_single)

# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
