import glob
import cv2
import numpy as np


def get_test():
    data = []
    labels = []

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test/crazing/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(0)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test/inclusion/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(1)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test/patches/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(2)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test/pitted_surface/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(3)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test/rolled_in_scale/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(4)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test/scratches/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(5)

    data = np.stack(data)  # array of shape [num_images, height, width, channel]
    labels = np.stack(labels)

    return data, labels

