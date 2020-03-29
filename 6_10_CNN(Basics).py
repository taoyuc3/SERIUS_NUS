import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Hyperparameters
# number of kernels in each convolution layer
numK = 16
# size of kernels in each convolution layer [nxn]
sizeConvK = 3
# size of kernels in each pool layer [mxm]
sizePoolK = 2
# size of the input image
inputSize = 28
# number of channels to the input image gray scale = 1, RGB=3
numChannels = 1

def convNet(inputs, labels, mode):
    # reshape the input from a vector to a 2D image
    input_layer = tf.reshape(inputs, [-1, inputSize, numChannels])

    # perform convolution and pooling
    conv1 = doConv(input_layer)
    pool1 = doConv(conv1)

    conv2 = doConv(pool1)
    pool2 = doPool(conv2)

    # flatted the result back to a vector for the FC flyer
    # images are 28 x 28 size, if put into 2 pooling layers it will half to (14 x 14)
    # half again of 7 x 7 in width
    # one 7 x 7 for each kernel, numk is because we will run more than one input image at a time
    #  -1 tells the TensorFlow to take all images and do the same to each (do this for the whole batch)
    flatPool = tf.reshape(pool2, [-1, 7 * 7 * numK])
    # we tell what to take as input, how many units we want it to have and what non-linearity we would prefer at the end
    # neat by typing all in a line
    dense = tf.layers.dense(inputs=flatPool, units=1024, activation=tf.nn.relu)

def doConv(inputs):
    convOut = tf.layers.conv2d(inputs=inputs, filters=numK, kernel_size=[sizeConvK, sizeConvK],
                               padding="SAME", activation=tf.nn.relu)
    return convOut

# strides=2 means we will half the size of the image at each pooling layer
def doPool(inputs):
    poolOut = tf.layers.max_pooling2d(inputs=inputs, pool_size=[sizePoolK, sizePoolK], strides=2)
    return poolOut

    # Get the output in the form of one-hot labels with x units
    # logits here is the outputs of the network corresponds to the 10 class of the training labels
    logits = tf.layers.dense(inputs=dense, unit=10)

    loss = None
    train_op = None
    # at the end of the network, check how well we did
    if mode != learn.ModeKeys.INFER:
        # tf.one_hot creates the one-hot labels from numeric training labels given to the network in labels
        # tf.cast operation make sure the data os of the correct type before doing the conversion
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        # check how close the output is to the training-labels
        # loss function takes in the output of the network logits and compares it to one_hot labels
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # after checking the loss, use it top train network weights
    # train_op here is a simple loss optimizer that tries to find the minimum loss for our data
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.cotrib.framework.get_global_step(),
                                                   learning_rate=learning_rate, optimizer="SGD")

    # show what the network has learned, so we output current predictions by defining a dictionary of data
    # the raw logits information and the associated probabilities (found by taking the softmax of the logits tensor)
    predictions = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits,
                                                                                              name="softmax_tensor")}
    # we can finish off our graph by making sure it returns data
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main(unused_argv):
    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# we create the classifier that will hold network and all of its data
# we tell it what our graph is called under model_fn and where we would like our output stored
mnistClassifier = learn.Estimator(model_fn=convNet,   model_dir="/tmp/mln_MNIST")

# we want to get some information out of our network that tells us about the training performance
# we create a dictionary that will hold the probabilities from the key that we named 'softmax_tenspr' in the graph
tensors2log = {"probabilities": "softmax_tensor"}
# how often we save this information is controlled with the every_n_iter attribute
logging_hook = tf.train.LoggingTensorHook(tensors=tensors2log, every_n_iter=100)

# Finally, TF trains the network
# we call .fit method of the classifier, passing the training data and the labels along with the batch size
# batch size: how much of the training data we want to use in each iteration
# we also want to monitor the training by outputting the data we've requested in logging_hook
mnistClassifier.fit(x=train_data, y=train_labels, batch_size=100, steps=1000, monitors=[logging_hook])
# when the training is complete, we'd like TF to take some test data and tell us how well the network performs
# we create metrics dictionary that TF will populate(i.e. enter/add data) by calling .evaluate
metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}

eval_results = mnistClassifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
print(eval_results)

if __name__ == "__main__":
    tf.app.run()
