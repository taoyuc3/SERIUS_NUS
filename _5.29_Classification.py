# https://www.tensorflow.org/tutorials/keras/basic_classification#explore_the_data
# TensorFlow and tf.keras
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import os
import h5py

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# explore the format of the dataset, and the we train the model
# train_images.shape
# # a pixel of 28x28 and 60,000 in general
# # train
# len(train_labels)
# train_labels
# test_images.shape
# test
len(test_labels)
# pre-process the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# we limit the data between 0-1, so we divide them by 255.0 (from integer to float)
train_images = train_images / 255.0
test_images = test_images / 255.0
# show the first 25 images and make sure the data is correct, and then we are able to construct networks
plt.figure(figsize=(10, 10))
# display the first 25 images and display the name below each image
# verify that data is in the correct form h5py and we are ready to build and train the network
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    # construct the model
    # set the layers
    # def create_model():
    model = keras.Sequential([
        # transfer the 2D 28x28 array to 1D
        keras.layers.Flatten(input_shape=(28, 28)),
        # the first dense layer has 128 neurons
        keras.layers.Dense(512, activation=tf.nn.relu),
        # drop put 20%
        keras.layers.Dropout(0.3),
        # the second layer
        keras.layers.Dense(512, activation=tf.nn.relu),
        # drop put 20%
        keras.layers.Dropout(0.3),
        #  the third layer
        keras.layers.Dense(512, activation=tf.nn.relu),
        # drop put 20%
        keras.layers.Dropout(0.3),
        # the softmax layer has 10 neurons,
        # this layer will return an array consists of 10 probability scores, which sum up to 1
        # each neuron contains one probability, representing one of 10 classifications
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              # loss function (make sure how accurate the model is during training)
              # optimizer (model updated based on data it sees and the loss function)
              # metrics (monitor training and test steps, now it uses for the fraction images are correctly classified)
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#     return model
# # create a basic model instance
#     model = create_model()
#     model.summary


# train the model
#  1. feed training data to the model
#  2. model learns to associate images and labels
#  3. after that we evaluate the predictions on the test set, verifying if images and labels would match
model.fit(train_images, train_labels, epochs=150, batch_size=10, verbose=1)
# save model to a HDF5 file
model.save("classification.h5")

# evaluate accuracy
# we compare how the model performs on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# make predictions
# when the model is trained, we can use it to make predictions about some images
predictions = model.predict(test_images)
# now model has predict the model for each image in the testing set
print(predictions[0])

print(np.argmax(predictions[0]))
plt.imshow(test_images[0])
