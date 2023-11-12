from sklearn.metrics import mean_squared_error, r2_score
import math
import constants
from ann_lightning import ANNLightning
from ann_vanilla import ANNVanilla


class AlgorithmRunner:
    @staticmethod
    def calculate_score(train_x, train_y,
                        test_x, test_y,
                        validation_x,
                        validation_y,
                        algorithm,
                        X_columns,
                        y_column
                        ):
        y_hats = None
        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
        if constants.LIGHTNING:
            ann = ANNLightning(algorithm, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column)
        else:
            ann = ANNVanilla(algorithm, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column)
        ann.train()
        y_hats = ann.test()
        r2 = r2_score(test_y, y_hats)
        rmse = math.sqrt(mean_squared_error(test_y, y_hats, squared=False))
        i,j = ann.get_model().get_params()
        return max(r2,0), rmse, i, j