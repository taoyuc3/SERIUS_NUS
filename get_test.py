import glob
import cv2
import numpy as np


def get_test():
    data = []
    labels = []
    img = []
    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test1/crazing/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(0)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test1/inclusion/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(1)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test1/patches/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(2)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test1/pitted_surface/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(3)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test1/rolled_in_scale/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(4)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/test1/scratches/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(5)

    # cv2.imshow('Original image', img)
    data = np.stack(data)  # array of shape [num_images, height, width, channel]
    labels = np.stack(labels)

    return data, labels

