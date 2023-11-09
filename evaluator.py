from reporter import Reporter
from algorithm_runner import AlgorithmRunner
from ds_manager import DSManager


class Evaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None, column_groups=None):
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms
        self.column_groups = column_groups

        if self.column_groups is None:
            self.column_groups = [[]]

        if self.algorithms is None:
            self.algorithms = ["mlr", "rf", "svr", "ann_simple", "ann_1dcnn1"]

        self.reporter = Reporter(prefix, self.column_groups, self.algorithms, self.repeat, self.folds)

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

    def process_repeat(self, repeat_number):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(repeat_number, index_algorithm)

    def process_algorithm(self, repeat_number, index_algorithm):
        for index_column_group in range(len(self.column_groups)):
            config = self.column_groups[index_column_group]
            print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]} - {Reporter.column_group_display(config)}")
            self.process_coolumn_group(repeat_number, index_algorithm, index_column_group)

    def process_coolumn_group(self, repeat_number, index_algorithm, index_column_group):
        algorithm = self.algorithms[index_algorithm]
        ds = DSManager(folds=self.folds, X_columns=self.column_groups[index_column_group])
        for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(ds.get_k_folds()):
            r2, rmse = self.reporter.get_details(index_algorithm, repeat_number, fold_number, index_column_group)
            if r2 != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                r2, rmse = AlgorithmRunner.calculate_score(train_x, train_y,
                                                           test_x, test_y,
                                                           validation_x, validation_y,
                                                           algorithm,
                                                           ds.X_columns,
                                                           ds.y_column,
                                                           )
            if self.verbose:
                print(f"{r2} - {rmse}")
                print(f"R2 - RMSE")
            self.reporter.set_details(index_algorithm, repeat_number, fold_number, index_column_group, r2, rmse)
            self.reporter.write_details()
            self.reporter.update_summary()

