import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
import constants
import spec_utils
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self, folds=10, X_columns=None, y_column="oc"):
        torch.manual_seed(0)
        df = pd.read_csv(constants.DATASET)
        if X_columns is None or len(X_columns) == 0:
            X_columns = spec_utils.get_wavelengths()

        self.X_columns = X_columns
        self.y_column = y_column
        self.folds = folds
        df = df[X_columns+[y_column]]
        df = df.sample(frac=1)
        self.full_data = df.to_numpy()
        self.full_data = DSManager._normalize(self.full_data)

    @staticmethod
    def _normalize(data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.full_data[test_index]
            train_x = train_data[:, :-1]
            train_y = train_data[:, -1]
            test_x = test_data[:, :-1]
            test_y = test_data[:, -1]
            validation_x = validation_data[:, :-1]
            validation_y = validation_data[:, -1]

            yield train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_folds(self):
        return self.folds

    def split_X_y(self):
        return DSManager.split_X_y_array(self.full_data)

    @staticmethod
    def split_X_y_array(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    def get_input_size(self):
        x, y = self.split_X_y()
        return x.shape[1]

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        return train_data, test_data
