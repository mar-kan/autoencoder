import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers, losses, metrics
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import BatchNormalization, MaxPooling1D, Conv1D, UpSampling1D


def encoder(num_of_convs, num_of_filters, conv_filter_size, input_img):
    # encoder with <num_of_convs> layers

    # creates 1st convolutional layer
    conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(input_img)
    conv = BatchNormalization()(conv)
    conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling1D(pool_size=2)(conv)

    # creates the rest layers
    i = 0
    while i < int(num_of_convs) - 1:
        if i == 0 or i == 1:
            conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(pool)
        else:
            conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(conv)

        conv = BatchNormalization()(conv)
        conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)

        # max pooling only after 1st and 2nd layers
        if i == 0:
            pool = MaxPooling1D(pool_size=2)(conv)

        num_of_filters *= 2
        i += 1

    return conv, num_of_filters