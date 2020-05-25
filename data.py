import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
from collections import defaultdict
from sklearn.preprocessing import normalize


class DataSet(object):
    def __init__(self, dynamic_features):
        self._dynamic_features = dynamic_features
        self._num_examples = dynamic_features.shape[0]
        self._epoch_completed = 0
        self._batch_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        if batch_size > self._num_examples or batch_size <=0:
            batch_size = self._dynamic_features.shape[0]
        if self._batch_completed ==0:
            self._shuffle()
        self._batch_completed += 1
        start = self._index_in_epoch
        if start + batch_size >= self._num_examples:
            self._epoch_completed += 1
            dynamic_rest_part = self._dynamic_features[start:self._num_examples]
            self._shuffle()
            self._index_in_epoch = 0
            return dynamic_rest_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_features[start:end]

    def predict_next_batch(self,batch_size):
        if batch_size > self._num_examples or batch_size <=0:
            batch_size = self._dynamic_features.shape[0]
        self._batch_completed += 1
        start = self._index_in_epoch
        if start + batch_size >= self._num_examples:
            self._epoch_completed += 1
            dynamic_rest_part = self._dynamic_features[start:self._num_examples]
            self._index_in_epoch = 0
            return dynamic_rest_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._dynamic_features[start:end]

    def _shuffle(self):
        index = np.arange(self._num_examples)
        np.random.shuffle(index)
        self._dynamic_features = self._dynamic_features[index]

    @property
    def dynamic_features(self):
        return self._dynamic_features

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @property
    def batch_completed(self):
        return self._batch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


def read_data():
    dynamic_features = np.load("../Trajectory_generate/Result/6_visit_format_normal.npy")
    feature = dynamic_features.reshape(-1, 30)
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_normalization = scaler.transform(feature)
    feature_normalization = feature_normalization.reshape(-1, 6, 30)
    return DataSet(feature_normalization)


def read_gaucoma_data():
    dynamic_features = np.load('patients_data_od.npy')[:, :, 2:].astype(float)  # OD data
    feature = dynamic_features.reshape(-1, dynamic_features.shape[2])
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_normalization = scaler.transform(feature)
    feature_normalization = feature_normalization.reshape(-1, 6, dynamic_features.shape[2])
    feature_normalization = np.concatenate((np.load('patients_data_od.npy')[:, :, 1].reshape(-1, 6, 1).astype(float), feature_normalization), axis=2)
    return DataSet(feature_normalization)


if __name__ == '__main__':
    read_gaucoma_data()
    # data = read_data()
