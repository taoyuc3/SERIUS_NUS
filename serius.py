from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FuncFormatter
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np
import get_data
import get_test
import random
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# parameters for architecture
random.seed(a=None, version=2)
input_shape = (200, 200, 1)
conv_size = 32
num_classes = 6

# parameters for training
batch_size = 32
num_epochs = 100

class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled_In_Scale', 'Scratches']

# build the model
model = Sequential()

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_train, y_train = get_data.get_data()
x_train = x_train / 255.0
print(x_train.shape, y_train.shape)
y_train = to_categorical(y_train, 6)

model.summary()

start = time.time()
# train the model
history = model.fit(x_train.reshape(-1, 200, 200, 1),
                    y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    validation_split=0.1)

end = time.time()

np.save("serius_history_acc", history.history['acc'])
np.save("serius_history_val_acc", history.history['val_acc'])

model.save("serius.h5")

x_test, y_test = get_test.get_test()
x_test = x_test / 255.0
y_test = to_categorical(y_test, 6)

print('\nDimension of images for testing is:', x_test.shape,
      '\nDimension of labels for testing is:', y_test.shape)

test_loss, test_acc = model.evaluate(x_test.reshape(-1, 200, 200, 1), y_test)

np.save("serius_history_loss", history.history['loss'])
np.save("serius_history_val_loss", history.history['val_loss'])

# printing functions
print("\nThe runtime is:", end - start, "seconds")
print('Test accuracy:', test_acc*100, "%")
print('──────────────────────────────────────────────────────────────')
j = input("Please enter a number (0-359): ")
i = int(j)

predictions = model.predict(x_test.reshape(-1, 200, 200, 1))

print("The prediction is:", np.argmax(predictions[i]), "―", class_names[int(np.argmax(predictions[i]))])
print("The actual label is:", y_test[i], "―", class_names[int(np.argmax(y_test[i]))], "\n")

print('The confusion matrix for testing:')
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))

s1 = class_names[int(np.argmax(predictions[i]))]
s2 = class_names[int(np.argmax(y_test[i]))]

# estimate if the prediction is correct
if s1 == s2:
    print('\nCongratulations, the prediction is correct.')
else:
    print('\nSorry, the prediction is not correct.')

# retrieve defect image and show
plt.figure(4)
plt.imshow(x_test[i,:,:], cmap='gray')
plt.title('Predicted Defect:' + s1 + '\nActual Defect:' + s2)
plt.axis('off')

# summarize history for accuracy
plt.figure(3)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()


def to_percent(temp, position):
    return '%1.0f' % (100*temp) + '%'


plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='upper left')

# confusion matrix heat map
plt.figure(1)
array = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
df_cm = pd.DataFrame(array, index=['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled_In', 'Scratches'],
                     columns=['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches'], )
sn.set(font_scale=1)
sn.heatmap(df_cm, linewidths=0.05, annot=True, annot_kws={"size": 10}, fmt="d")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

