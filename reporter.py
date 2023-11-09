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
        self.details_columns = self.get_details_columns()
        self.summary_columns = self.get_summary_columns()
        self.details_text_columns = ["algorithm", "column_group"]
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.details = np.zeros((len(self.algorithms) * len(self.column_groups), self.repeat * self.folds * 2))
        self.sync_details_file()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=self.details_text_columns, axis=1, inplace=True)
        self.details = df.to_numpy()

    @staticmethod
    def column_group_display(column_group):
        sorted_colgroup = sorted(column_group)
        if sorted(spec_utils.get_rgb()) == sorted_colgroup:
            return "rgb"
        if sorted(spec_utils.get_wavelengths()) == sorted_colgroup:
            return "bands"
        if len(column_group) == 0:
            return "all"
        name = "-".join(column_group[0:3])
        if len(column_group) > 3:
            name = f"{name}+{len(column_group)-3}"
        return name

    @staticmethod
    def column_groups_display(column_groups):
        return [Reporter.column_group_display(cg) for cg in column_groups]

    def write_summary(self, summary):
        summary_copy = np.round(summary,3)
        df = pd.DataFrame(data=summary_copy, columns=self.summary_columns)
        df.insert(0, "column_group", pd.Series(Reporter.column_groups_display(self.column_groups)))
        df.to_csv(self.summary_file, index=False)

    def find_mean_of_done_iterations(self, detail_cells):
        detail_cells = detail_cells[detail_cells != 0]
        if len(detail_cells) == 0:
            return 0
        else:
            return np.mean(detail_cells)

    def update_summary(self):
        score_mean = np.zeros((len(self.column_groups), 2 * len(self.algorithms)))
        iterations = self.repeat * self.folds
        for index_column_group in range(len(self.column_groups)):
            for index_algorithm in range(len(self.algorithms)):
                details_row = self.get_details_row(index_algorithm, index_column_group)
                detail_r2_cells = self.details[details_row, 0:iterations]
                r2_column_index = index_algorithm
                score_mean[index_column_group, r2_column_index] = self.find_mean_of_done_iterations(detail_r2_cells)
                detail_rmse_cells = self.details[details_row, iterations:]
                rmse_column_index = len(self.algorithms) + index_algorithm
                score_mean[index_column_group, rmse_column_index] = self.find_mean_of_done_iterations(detail_rmse_cells)
        self.write_summary(score_mean)

    def get_details_alg_column_group(self):
        details_alg_column_group = []
        for i in self.algorithms:
            for j in self.column_groups:
                details_alg_column_group.append((i,j))
        return details_alg_column_group

    def get_details_row(self, index_algorithm, index_column_group):
        return index_algorithm*len(self.column_groups) + index_column_group

    def get_details_column(self, repeat_number, fold_number, metric):
        #metric: 0,1: r2, rmse
        return (metric * self.repeat * self.folds ) + (repeat_number*self.folds + fold_number)

    def set_details(self, index_algorithm, repeat_number, fold_number, index_column_group, r2, rmse):
        details_row = self.get_details_row(index_algorithm, index_column_group)
        details_column_r2 = self.get_details_column(repeat_number, fold_number, 0)
        details_column_rmse = self.get_details_column(repeat_number, fold_number, 1)
        self.details[details_row, details_column_r2] = r2
        self.details[details_row, details_column_rmse] = rmse

    def get_details(self, index_algorithm, repeat_number, fold_number, index_column_group):
        details_row = self.get_details_row(index_algorithm, index_column_group)
        details_column_r2 = self.get_details_column(repeat_number, fold_number, 0)
        details_column_rmse = self.get_details_column(repeat_number, fold_number, 1)
        return self.details[details_row,details_column_r2], self.details[details_row,details_column_rmse]

    def get_details_columns(self):
        cols = []
        for metric in ["R2", "RMSE"]:
            for repeat in range(1,self.repeat+1):
                for fold in range(1,self.folds+1):
                    cols.append(f"{metric}({repeat}-{fold})")
        return cols

    def get_summary_columns(self):
        cols = []
        for metric in ["R2", "RMSE"]:
            for algorithm in self.algorithms:
                cols.append(f"{metric}({algorithm})")
        return cols

    def write_details(self):
        details_copy = np.round(self.details, 3)
        df = pd.DataFrame(data=details_copy, columns=self.details_columns)
        details_alg_conf = self.get_details_alg_column_group()
        algs = [i[0] for i in details_alg_conf]
        col_groups = [i[1] for i in details_alg_conf]

        df.insert(0,"column_group",pd.Series(Reporter.column_groups_display(col_groups)))
        df.insert(0,"algorithm",pd.Series(algs))

        df.to_csv(self.details_file, index=False)
