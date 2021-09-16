import numpy as np


def loadDataset(full_path):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(full_path, dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

    return data


def loadLabelSet(full_path):
    intType = np.dtype('int32').newbyteorder('>')
    labels = np.fromfile(full_path, dtype='ubyte')[2 * intType.itemsize:]

    return labels