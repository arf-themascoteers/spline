import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
import constants
import spec_utils


class DSManager:
    def __init__(self, folds=10, X_columns=None, y_column = "oc"):
        cols_to_remove = ["id", "lc1", "lu1"]
        cols_cat = ["lc1", "lu1"]
        cols_props = ["phh","phc","ec","caco3","p","n","k","elevation","stones","oc"]
        cols_wavelengths = spec_utils.get_wavelengths()

        torch.manual_seed(0)
        df = pd.read_csv(constants.DATASET)
        if X_columns is None or len(X_columns)==0:
            all_columns = list(df.columns)
            for c in cols_to_remove:
                all_columns.remove(c)
            X_columns = all_columns
            X_columns.remove(y_column)

        self.X_columns = X_columns
        self.y_column = y_column
        self.folds = folds

        base_columns = []
        derived_columns = []

        for c in self.X_columns:
            if c in df.columns:
                base_columns.append(c)
            else:
                derived_columns.append(c)

        df2 = df[base_columns]
        df2 = df2.sample(frac=1)

        for dc in derived_columns:
            df2[dc] = self.derive(df, dc)

        df2[self.y_column] = df[self.y_column]
        self.full_data = df2.to_numpy()

    def derive(self, df, si):
        if si == "savi":
            L = 0.5
            nir = df["vnir4"]
            red = df["red"]
            return (nir - red)/(nir + red + L)*(1+L)
        elif si == "savi2":
            L = 35
            nir = df["vnir4"]
            red = df["red"]
            return (nir - red)/(nir + red + L)*(1+L)
        return None


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
        return x,y

    def get_input_size(self):
        x,y = self.split_X_y()
        return x.shape[1]

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        return train_data, test_data
