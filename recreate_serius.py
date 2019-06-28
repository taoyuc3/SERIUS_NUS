from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FuncFormatter
from keras.utils import to_categorical
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np
import get_data
import get_test

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']

x_train, y_train = get_data.get_data()
x_train = x_train / 255.0
# print(x_train.shape, y_train.shape)
y_train = to_categorical(y_train, 6)

x_test, y_test = get_test.get_test()
x_test = x_test / 255.0
y_test = to_categorical(y_test, 6)
# print('\nThe dimension of images for testing is:', x_test.shape,
#       '\nThe dimension of labels for testing is:', y_test.shape)

new_model = keras.models.load_model('best2.h5')

# new_model.summary()

predictions = new_model.predict(x_test.reshape(-1, 200, 200, 1))

# print('\nThe confusion matrix for testing is:')
# print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))
# while 1:
# print(confusion_matrix(y_test, predictions))

j = input("\nPlease enter a number (0-359): ")
i = int(j)
# printing stuffs

print("\nThe prediction is:", np.argmax(predictions[i]), "―", class_names[int(np.argmax(predictions[i]))])
print("The actual label is:", y_test[i], "―", class_names[int(np.argmax(y_test[i]))])

s1 = class_names[int(np.argmax(predictions[i]))]
s2 = class_names[int(np.argmax(y_test[i]))]

if s1 == s2:
    print('\nThe prediction is correct.')
else:
    print('\nThe prediction is not correct.')

# plot confusion matrix
plt.figure(2)
plt.imshow(x_test[i, :, :], cmap='gray')
plt.title('Prediction: ' + s1 + '\nActual: ' + s2)
plt.axis('off')

# # summarize history for accuracy
# plt.figure(3)
# acc = np.load('serius_history_acc.npy')
# val_acc = np.load('serius_history_val_acc.npy')
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.grid()
#
#
# def to_percent(temp):
#     return '%1.0f' % (100 * temp) + '%'
#
#
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#
# plt.legend(['train', 'test'], loc='upper left')
#
# # summarize history for loss
# plt.figure(2)
# loss = np.load('serius_history_loss.npy')
# val_loss = np.load('serius_history_val_loss.npy')
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.grid()
# plt.legend(['train', 'test'], loc='upper left')

# plot confusion matrix
plt.figure(1)
array = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
df_cm = pd.DataFrame(array, index=['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'],
                     columns=['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'], )
sn.set(font_scale=1)
sn.heatmap(df_cm, linewidths=0.5, annot=True, annot_kws={"size": 13}, fmt="d")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
