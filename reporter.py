import numpy as np
import os
import pandas as pd
import spec_utils


class Reporter:
    def __init__(self, prefix, column_groups, algorithms, repeat, folds):
        self.prefix = prefix
        self.column_groups = column_groups
        self.algorithms = algorithms
        self.repeat = repeat
        self.folds = folds
        self.metrics = ["R2", "RMSE", "i", "j"]
        self.details_columns = self.get_details_columns()
        self.details_text_columns = ["repeat", "fold"]
        self.details_file = f"results/{prefix}_details.csv"
        self.details = np.zeros((self.get_count_iterations(), len(self.metrics)))
        self.sync_details_file()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=self.details_text_columns, axis=1, inplace=True)
        self.details = df.to_numpy()

    def fold_display(self):
        folds = []
        for r in range(self.repeat):
            folds = folds + [f"{f}" for f in range(self.folds)]
        return folds

    def repeat_display(self):
        repeats = []
        for r in range(self.repeat):
            repeats = repeats + [f"{r}" for f in range(self.folds)]
        return repeats

    def get_count_iterations(self):
        return self.repeat*self.folds

    def get_details_row(self, repeat_number, fold_number):
        return repeat_number*self.folds + fold_number

    def get_details_column(self, metric):
        return self.metrics.index(metric)

    def set_details(self, repeat_number, fold_number, r2, rmse, i, j):
        details_row = self.get_details_row(repeat_number, fold_number)
        details_column_r2 = self.get_details_column("R2")
        details_column_rmse = self.get_details_column("RMSE")
        details_column_i = self.get_details_column("i")
        details_column_j = self.get_details_column("j")
        self.details[details_row, details_column_r2] = r2
        self.details[details_row, details_column_rmse] = rmse
        self.details[details_row, details_column_i] = i
        self.details[details_row, details_column_j] = j

    def get_details(self, repeat_number, fold_number):
        details_row = self.get_details_row(repeat_number, fold_number)
        details_column_r2 = self.get_details_column("R2")
        details_column_rmse = self.get_details_column("RMSE")
        details_column_i = self.get_details_column("i")
        details_column_j = self.get_details_column("j")
        return (self.details[details_row,details_column_r2], self.details[details_row,details_column_rmse],
                self.details[details_column_i, self.details[details_column_j]])

    def get_details_columns(self):
        return self.metrics

    def write_details(self):
        details_copy = np.round(self.details, 3)
        df = pd.DataFrame(data=details_copy, columns=self.details_columns)
        df.insert(0,"fold", pd.Series(self.fold_display()))
        df.insert(0,"repeat",pd.Series(self.repeat_display()))
        df.to_csv(self.details_file, index=False)
