import pandas as pd
import torch
from sklearn import model_selection
import constants
import spec_utils
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self):
        torch.manual_seed(0)
        df = pd.read_csv(constants.DATASET)
        self.X_columns = spec_utils.get_wavelengths()
        self.y_column = "oc"
        df = df[self.X_columns+[self.y_column]]
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

    def get_datasets(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        validation_x = validation_data[:, :-1]
        validation_y = validation_data[:, -1]

        return train_x, train_y, test_x, test_y, validation_x, validation_y

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
