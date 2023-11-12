from reporter import Reporter
from algorithm_runner import AlgorithmRunner
from ds_manager import DSManager
import spec_utils


class Evaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithm=None, column_group=None):
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithm = algorithm
        self.column_group = column_group

        if self.column_group is None:
            self.column_group = spec_utils.get_wavelengths()

        if self.algorithm is None:
            self.algorithm = "ann_simple"

        self.reporter = Reporter(prefix, self.column_group, self.algorithm, self.repeat, self.folds)

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

    def process_repeat(self, repeat_number):
        ds = DSManager(folds=self.folds, X_columns=self.column_group)
        for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(ds.get_k_folds()):
            r2, rmse,i,j = self.reporter.get_details(repeat_number, fold_number)
            if r2 != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                r2, rmse, i, j = AlgorithmRunner.calculate_score(train_x, train_y,
                                                           test_x, test_y,
                                                           validation_x, validation_y,
                                                           self.algorithm,
                                                           ds.X_columns,
                                                           ds.y_column,
                                                           )
            if self.verbose:
                print(f"{r2} - {rmse} - {i} - {j}")
            self.reporter.set_details(repeat_number, fold_number, r2, rmse, i, j)
            self.reporter.write_details()

