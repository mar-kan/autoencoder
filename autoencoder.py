import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers, losses, metrics
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import BatchNormalization, MaxPooling1D, Conv1D, UpSampling1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from keras.models import load_model
from encoder import encoder
from decoder import decoder
from loadDataset import loadDataset


def main():
    # checking arguments
    if len(sys.argv) < 3 or sys.argv[1] != '-d':
        print('Wrong arguments')
        exit(-1)

    # setting new path where dataset is
    path = sys.argv[2]

    # reading train set
    trainImages = loadDataset(path)

    # normalization
    X_train = trainImages / 255.0

    # reshape data to 1D vectors
    X_train = X_train.reshape(60000, 784)

    # create input channel
    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape=(x * y, inChannel))

    answer = 'yes'
    while answer == 'yes':
        #user inputs hyperparameters
        print()
        print('Input number of convolutional layers')
        num_of_convs = int(input())
        print('Input size of convolutional filters')
        conv_filter_size = int(input())
        print('Input number of convolutional filters per layer')
        num_of_filters = int(input())
        print('Input number of epochs')
        epochs = int(input())
        print('Input batch size')
        batch_size = int(input())

        #nn is created
        autoencoder = Model(input_img,
                            decoder(encoder(num_of_convs, num_of_filters, conv_filter_size, input_img), num_of_convs,
                                    conv_filter_size))
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

        train_X, valid_X, train_ground, valid_ground = train_test_split(X_train, X_train, test_size=0.2,
                                                                        random_state=13)

        autoencoder_history = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1,
                                              validation_data=(valid_X, valid_ground))

        # user's choices
        print('Would you like to present the loss plots for these hyperparameters?')
        print('(Type yes or no)')
        if input() == 'yes':
            # plotting loss function
            ax = plt.subplot(1, 1, 1)
            ax.plot(autoencoder_history.history['loss'], 'b', label='Loss for batch size: ' + str(batch_size))
            ax.plot(autoencoder_history.history['val_loss'], 'g', label='Validation Loss for batch size: '
                                                                        + str(batch_size))
            ax.set_xlabel(r'Epoch')
            ax.set_ylabel(r'Loss')
            ax.legend(loc='upper right')
            plt.show()

        print('Would you like to save the model with these hyperparameters?')
        print('(Type yes or no)')
        if input() == 'yes':
            # saving model in current directory
            autoencoder.save(os.getcwd() + '/autoencoder.h5', overwrite=True)
            autoencoder.summary()

        print('Would you like to repeat the process with different hyperparameters?')
        print('(Type yes or no)')
        answer = input()


if __name__ == "__main__":
    # device_name = tf.test.gpu_device_name()
    # if device_name != '/device:GPU:0':
    #    raise SystemError('GPU not found')
    # print('GPU at : {}'.format(device_name))
    main()
