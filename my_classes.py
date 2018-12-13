#!/usr/bin/env python3

#
#   Example keras data generator from:
#
#   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
#

import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, augmentation, scale_image, batch_size=32,
                 dim=(299,299), n_channels=3, n_classes=100, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.augmentation = augmentation
        self.scale_image = scale_image
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.img_prefix = '../data/boxed/'
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start_index = index*self.batch_size
        end_index = (index+1)*self.batch_size
        indexes = self.indexes[start_index:end_index]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def augment_image(self, x):
        val = np.random.randint(0, 4)
        if val == 0:
            x = image.random_brightness(x, (0.01, 0.99))
        elif val == 1:
            x = image.random_rotation(x, 20, row_axis=0, col_axis=1, channel_axis=2)
        elif val == 2:
            x = image.random_shift(x, 0.1, 0.1, row_axis=0, col_axis=1, channel_axis=2) 
        elif val == 3:
            x = image.random_zoom(x, (0.1, 0.1), row_axis=0, col_axis=1, channel_axis=2) 
        elif val == 4:
            x = image.random_shear(x, 30, row_axis=0, col_axis=1, channel_axis=2) 
        else:
            assert False, 'Fatal error, DataGenerator.augment_image(), val out of range'
        return x    


    def get_image(self, image_id):
        'loads single image from disk, scales it and returns it as numpy array'
        img_path = self.img_prefix + image_id + '.jpg'
        img = image.load_img(img_path, target_size=self.dim)
        x = image.img_to_array(img)
        if self.augmentation:
            x = self.augment_image(x)
        x = self.scale_image(x)
        x = np.expand_dims(x, axis=0)
        img.close()
        return x

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # get image data
            X[i,] = self.get_image(ID)

            # store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

