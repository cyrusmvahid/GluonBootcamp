import numpy as np
from mxnet import gluon
import os.path

class TimeSeriesData(object):
    """
    Reads data from file and creates training and validation datasets
    """
    def __init__(self, file_path, window, horizon, train_ratio=1.0):
        """
        :param str file_path: path to the data file (e.g. electricity.txt)
        """
        print('Loading file {}'.format(file_path))
        print('Is it a file {}'.format(os.path.isfile(file_path)))
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        train_data_len = int(len(data) * train_ratio)
        print('Data length {}'.format(train_data_len))
        self.num_series = data.shape[1]
        if train_ratio > 0.0:
            self.train = TimeSeriesDataset(data[:train_data_len], window=window, horizon=horizon)
        if train_ratio < 1.0:
            self.val = TimeSeriesDataset(data[train_data_len:], window=window, horizon=horizon)


class TimeSeriesDataset(gluon.data.Dataset):
    """
    Dataset that splits the data into a dense overlapping windows
    """
    def __init__(self, data, window, horizon, transform=None):
        """
        :param np.ndarray data: time-series data in TC layout (T: sequence len, C: channels)
        :param int window: context window size
        :param int horizon: prediction horizon
        :param function transform: data transformation function: fn(data, label)
        """
        super(TimeSeriesDataset, self).__init__()
        self._data = data
        self._window = window
        self._horizon = horizon
        self._transform = transform

    def __getitem__(self, idx):
        """
        :param int idx: index of the item
        :return: single item in 'TC' layout
        :rtype np.ndarray
        """
        assert idx < len(self)
        data = self._data[idx:idx + self._window]
        label = self._data[idx + self._window + self._horizon - 1]
        if self._transform is not None:
            return self._transform(data, label)
        return data, label

    def __len__(self):
        """
        :return: length of the dataset
        :rtype int
        """
        return len(self._data) - self._window - self._horizon
