from reporter import Reporter
from ds_manager import DSManager
from sklearn.metrics import mean_squared_error, r2_score
import math
import constants
from ann_lightning import ANNLightning
from ann_vanilla import ANNVanilla


class Evaluator:
    def __init__(self, name="", verbose=False):
        self.verbose = verbose
        self.reporter = Reporter(name)

    def process(self):
        ds = DSManager()
        train_x, train_y, test_x, test_y, validation_x, validation_y = ds.get_datasets()
        y_hats = None
        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
        if constants.LIGHTNING:
            ann = ANNLightning(train_x, train_y, test_x, test_y, validation_x, validation_y, self.reporter)
        else:
            ann = ANNVanilla(train_x, train_y, test_x, test_y, validation_x, validation_y, self.reporter)
        ann.train()
        y_hats = ann.test()
        r2 = r2_score(test_y, y_hats)
        rmse = math.sqrt(mean_squared_error(test_y, y_hats, squared=False))
        return max(r2,0), rmse

