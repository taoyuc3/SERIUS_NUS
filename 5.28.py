from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
# the dataset has 9 cols, 0:8 will select 0 to 7, stopping before index 8
# *Numpy Arrays for ML in Python
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Define a sequential model and add layers one at a time
model = Sequential()
# first layer has 100 neurons and expects 8 inputs with rectifier activation function
model.add(Dense(100, input_dim=8, activation='relu'))
# the first hidden layer has 60 neurons
model.add(Dense(100, activation='relu'))
# the second hidden layer has 80 neurons
model.add(Dense(100, activation='relu'))
# the third hidden layer has 100 neurons
model.add(Dense(100, activation='relu'))
# finally the output layer has 1 neuron to predict the class (onset of diabetes or not)
model.add(Dense(1, activation='sigmoid'))

# Compile modelsurface-defect-detection
# binary_crossentropy refer to logarithmic loss
# adam refers to gradient descent
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
# 150 iterations with the batch size 10
# these can be chosen experimentally by trial and error
model.fit(X, Y, epochs=1500, batch_size=10)

# Evaluate
# we have trained our nn on the entire dataset, but only know train accuracy
# don't know how well the algorithm might perform on new data
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
