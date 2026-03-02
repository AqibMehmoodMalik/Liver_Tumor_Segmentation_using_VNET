import h5py
import matplotlib.pyplot as plt


def print_slice(data, truth, slice):
    fig = plt.figure()
    y = fig.add_subplot(1, 2, 1)
    y.imshow(data[:, :, slice, 0], cmap='gray')
    y.set_title('Volume')
    y.set_xlabel('x axis')
    y.set_ylabel('y axis')
    y = fig.add_subplot(1, 2, 2)
    # y.imshow(truth[:, :, slice, 0]*data[:, :, slice, 0], cmap='gray')
    y.imshow(truth[:, :, slice, 0], cmap='gray')
    y.set_title('Segmentation')
    y.set_xlabel('x axis')
    y.set_ylabel('y axis')
    plt.show


def load_dataset(path, h5=True):
    #print("Loading dataset... Shape:")
    file = h5py.File(path, 'r')
    if h5:
        data = file.get('data')
        truth = file.get('truth')
    else:
        data = file.get('data').value
        truth = file.get('truth').value
    # print(data.shape)
    return data, truth

# utils.py (add this)
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

class HDF5BatchGenerator(Sequence):
    def __init__(self, h5_path, batch_size=1, shuffle=False):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Open once to read shapes
        with h5py.File(self.h5_path, "r") as f:
            self.x_shape = f["data"].shape[0]

        self.indices = np.arange(self.x_shape)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.x_shape / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        with h5py.File(self.h5_path, "r") as f:
            X = f["data"][batch_idx]
            Y = f["truth"][batch_idx]
        if Y.ndim == 5:
          Y = np.squeeze(Y, axis=-1)
        Y = tf.one_hot(Y.astype(np.uint8), depth=3)
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
