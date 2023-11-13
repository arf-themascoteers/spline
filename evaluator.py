from reporter import Reporter
from algorithm_runner import AlgorithmRunner
from ds_manager import DSManager
import spec_utils


class Evaluator:
    def __init__(self, prefix="", verbose=False):
        self.verbose = verbose
        self.reporter = Reporter(prefix, self.repeat, self.folds)

    def process(self):
        ds = DSManager()
        for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(ds.get_datasets()):
            r2, rmse,i,j = self.reporter.get_details(repeat_number, fold_number)
            if r2 != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                r2, rmse, i, j = AlgorithmRunner.calculate_score(train_x, train_y,
                                                           test_x, test_y,
                                                           validation_x, validation_y,
                                                           ds.X_columns,
                                                           ds.y_column,
                                                           )
            if self.verbose:
                print(f"{r2} - {rmse} - {i} - {j}")
            self.reporter.set_details(repeat_number, fold_number, r2, rmse, i, j)
            self.reporter.write_details()

