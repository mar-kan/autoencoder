import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers, losses, metrics
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import BatchNormalization, MaxPooling1D, Conv1D, UpSampling1D


def decoder(encoder_results, num_of_convs, conv_filter_size):
    # decoder with <num_of_convs> layers

    conv = encoder_results[0]
    num_of_filters = encoder_results[1] / 2

    # creates 1st convolutional layer
    conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)

    # creates the rest layers
    i = 0
    while i < num_of_convs - 2:
        if i == num_of_convs - 3 and i != 0:
            conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(up)
        else:
            conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(conv)

        conv = BatchNormalization()(conv)
        conv = Conv1D(num_of_filters, conv_filter_size, activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)

        # up sampling only in last and 2nd from last layers
        if i - num_of_convs - 4 <= 0:
            up = UpSampling1D(2)(conv)

        num_of_filters /= 2
        i += 1

    if num_of_convs == 2:
        up = UpSampling1D(2)(conv)

    # creates last layer
    decoded = Conv1D(1, conv_filter_size, activation='sigmoid', padding='same')(up)
    return decoded
