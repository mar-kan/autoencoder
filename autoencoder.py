import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers, losses, metrics
from tensorflow import keras

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#    raise SystemError('GPU not found')
# print('GPU at : {}'.format(device_name))

# read test set for later
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import BatchNormalization, MaxPooling1D, Conv1D, UpSampling1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop


def encoder(input_img):
    # encoder
    # #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_img)  # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)  # 14 x 14 x 32
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(pool1)  # 14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)  # 7 x 7 x 64
    conv3 = Conv1D(128, 3, activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small&thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(128, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(256, 3, activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small& thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(256, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4


def decoder(conv4):
    # decoder
    conv5 = Conv1D(128, 3, activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(64, 3, activation='relu', padding='same')(conv5)  # 7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(64, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling1D(2)(conv6)  # 14 x 14 x 64
    conv7 = Conv1D(32, 3, activation='relu', padding='same')(up1)  # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(32, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling1D(2)(conv7)  # 28 x 28 x 32
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
    return decoded


def loadDataset(prefix, full_path):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(full_path + prefix + '-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

    labels = np.fromfile(full_path + prefix + '-labels-idx1-ubyte',
                         dtype='ubyte')[2 * intType.itemsize:]

    return data, labels


def main():
    # checking arguments
    if len(sys.argv) < 3 or sys.argv[1] != '-d':
        print('Wrong arguments')
        exit(-1)

    # setting new path where dataset is
    path = sys.argv[2]

    # reading train set
    trainImages, trainLabels = loadDataset("train", os.getcwd() + "/datasets2/")
    # testImages, testLabels = loadDataset("t10k", os.getcwd() + "/datasets2/")

    # normalization
    X_train = trainImages / 255.0
    Y_train = trainLabels / 255.0

    # reshape data to 1D vectors
    X_train = X_train.reshape(60000, 784)
    # X_test = X_test.reshape(10000, 784)

    # Convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, 10)
    # Y_test = keras.utils.to_categorical(Y_test, 10)

    epochs = 60
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8

    # Neural network
    # building model
    #model = keras.Sequential()
    # Adds a densely-connected layer with 64 nodes to the model:
    # input shape defines the number of input features (dimensions)
    # activation defines the activation function of each layer
    #model.add(layers.Dense(64, activation='relu'))
    # Add another:
    #model.add(layers.Dense(64, activation='relu'))
    # Add a softmaxlayer with 10 output nodes:
    #model.add(layers.Dense(10, activation='softmax'))

    # compile the model
    #model.compile(optimizer=optimizers.RMSprop(0.01), loss=losses.CategoricalCrossentropy(),
                  #metrics=[metrics.CategoricalAccuracy()])

    # training model
    # batch_size = int(X_train.shape[1] / 100)

    # model_history = model.fit(X_train, Y_train,
    # batch_size=batch_size,
    # epochs=epochs)

    # data = np.random.random((1000, 10))
    # labels = np.random.random((1000, 10))
    #model_history = model.fit(X_train, Y_train, epochs=10, batch_size=100)

    # evaluating model
    #model.evaluate(X_train, Y_train, batch_size=32)

    batch_size = 128
    epochs = 50
    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape=(x*y, inChannel), activation='softmax')

    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

    train_X, valid_X, train_ground, valid_ground = train_test_split(X_train,
                                                                    X_train,
                                                                    test_size=0.2,
                                                                    random_state=13)

    autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1,
                                        validation_data=(valid_X, valid_ground))

    # saving model in current directory
    autoencoder.save_weights(os.getcwd()+'autoencoder.h5')
    autoencoder.summary()
    autoencoder.layers[0].get_weights()

    # Plot the loss function
    ax = plt.subplot(2, 1, 1)
    ax.plot(np.sqrt(autoencoder_train.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(autoencoder_train.history['val_loss']), 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    plt.legend()
    ax.tick_params(labelsize=20)
    plt.show()

    # Plot the accuracy
    ax = plt.subplot(2, 1, 2)
    ax.plot(np.sqrt(autoencoder_train.history['acc']), 'r', label='train')
    ax.plot(np.sqrt(autoencoder_train.history['val_acc']), 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Accuracy', fontsize=20)
    ax.legend()     # Wait for it.........  DARY legendary. Up top
    ax.tick_params(labelsize=20)
    plt.show()


if __name__ == "__main__":
    main()
