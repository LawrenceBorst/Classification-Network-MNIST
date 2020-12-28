import struct
from array import array
import numpy as np

class LoadData:
    def __init__(self, X_labels, X_images, Y_labels, Y_images):
        self.X_labels = X_labels
        self.X_images = X_images
        self.Y_labels = Y_labels
        self.Y_images = Y_images

    # The below get functions are mostly copied from Kaggle
    def get_train_data(self):
        with open(self.X_labels, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch.')
            labels = array("B", file.read())

        with open(self.X_images, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch.')
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype="float16")
            images[i] = img

        return labels, images

    def get_test_data(self):
        with open(self.Y_labels, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch.')
            labels = array("B", file.read())

        with open(self.Y_images, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch.')
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype="float16")
            images[i] = img

        return labels, images

    def preprocess(self, images):
        img_total = len(images)  # Total number of images
        img_length = len(images[0])  # Number of pixels in a single image
        mean = np.zeros(shape=img_length, dtype="float16")

        for i in range(img_total):
            images[i] = images[i] / 255  # Normalize pixel range
            mean += images[i]
        mean = mean / img_total

        for i in range(img_total):
            images[i] -= mean # Centering
        return images