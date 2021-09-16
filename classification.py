import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers, losses, metrics
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from keras.models import load_model
from loadDataset import loadDataset, loadLabelSet


def checkArguments():  # checks script's arguments
    if len(sys.argv) < 11:
        print('Too few arguments')
        exit(-1)

    # requires this specific order to run
    # –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>
    if sys.argv[1] != '-d' or sys.argv[3] != '-dl' or sys.argv[5] != '-t' or sys.argv[7] != '-tl' or sys.argv[
        9] != '-model':
        print('Wrong argument')
        exit(-2)

    training_set = sys.argv[2]
    training_labels = sys.argv[4]
    test_set = sys.argv[6]
    test_labels = sys.argv[8]
    model_name = sys.argv[10]

    return training_set, training_labels, test_set, test_labels, model_name


def main():
    training_set, training_labels, test_set, test_labels, model_name = checkArguments()

    # reading train set
    trainImages = loadDataset(training_set)
    trainLabels = loadLabelSet(training_labels)

    # reading test set
    testImages = loadDataset(test_set)
    testLabels = loadLabelSet(test_labels)

    # normalization
    X_train = trainImages / 255.0
    Y_train = trainLabels
    X_test = testImages / 255.0
    Y_test = testLabels

    # reshape data to 1D vectors
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # Convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    # set hyperparameters
    print('Input number of convolutional filters per layer')
    num_of_filters = int(input())
    print('Input number of epochs')
    epochs = int(input())
    print('Input batch size')
    batch_size = int(input())

    # loads the model of the autoencoder
    model = load_model(model_name)

    # counts previous layers and makes them untrainable
    count = 0
    for layer in model.layers:
        layer.trainable = False
        count += 1

    # summarize the structure of the model
    model.summary()

    # adds fully connected layer
    model.dense = layers.Dense(3, activation='relu')

    # adds output layer
    output = layers.Conv1D(1, num_of_filters, activation='softmax', padding='same')

    # create and compile new NN
    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape=(x * y, inChannel))

    new_model = Model(input_img, model, model.dense, output)
    new_model.compile(loss=losses.CategoricalCrossentropy, optimizer=RMSprop())

    train_X, valid_X, train_ground, valid_ground = train_test_split(X_train, X_train, test_size=0.2,
                                                                    random_state=13)
    # train new model
    history = new_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(valid_X, valid_ground))

    # (accuracy), σφάλμα(loss), ακρίβεια(precision), ανάκληση(recall), f - score.
    answer = 'yes'
    while answer == 'yes':
        print('Would you like to present the loss plots for these hyperparameters?')
        print('(Type yes or no)')
        if input() == 'yes':
            # plotting loss function
            ax = plt.subplot(1, 1, 1)
            #ax.plot(history.history['loss'], 'b', label='Loss for batch size: ' + str(batch_size))
            #ax.plot(history.history['val_loss'], 'g',
                    #label='Validation Loss for batch size: ' + str(batch_size))
            ax.set_xlabel(r'Epoch')
            ax.set_ylabel(r'Loss')
            ax.legend(loc='upper right')
            plt.show()

        print('Would you like to classify the test set?')
        print('(Type yes or no)')
        if input() == 'yes':
            # input hyperparameters
            print('Input number of epochs')
            epochs = int(input())
            print('Input batch size')
            batch_size = int(input())

            # evaluating model with test set
            new_model.evaluate(X_test, Y_test, batch_size=batch_size)
            print(model(X_test, Y_test))

        print('Would you like to repeat the process with different hyperparameters?')
        print('(Type yes or no)')
        answer = input()


if __name__ == "__main__":
    main()
