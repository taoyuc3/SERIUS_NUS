import glob
import cv2
import numpy as np


def get_data():
    data = []
    labels = []
    img = []
    for i in glob.glob('./NEU-DET/IMAGES_Dataset/train/crazing/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(0)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/train/inclusion/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(1)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/train/patches/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(2)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/train/pitted_surface/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(3)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/train/rolled_in_scale/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(4)

    for i in glob.glob('./NEU-DET/IMAGES_Dataset/train/scratches/*.jpg', recursive=True):
        img = cv2.imread(i)
        data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(5)

# cv2.imshow('Original image', img)
    data = np.stack(data)  # array of shape [num_images, height, width, channel]
    labels = np.stack(labels)

    return data, labels




