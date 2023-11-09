from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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
        if algorithm.startswith("ann_"):
            if constants.LIGHTNING:
                ann = ANNLightning(algorithm, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column)
            else:
                ann = ANNVanilla(algorithm, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column)
            ann.train()
            y_hats = ann.test()
            # if algorithm == "ann_savi":
            #     print(ann.model.L)
        else:
            model_instance = None
            if algorithm == "mlr":
                model_instance = LinearRegression()
            elif algorithm == "plsr":
                size = train_x.shape[1]//2
                if size == 0:
                    size = 1
                model_instance = PLSRegression(n_components=size)
            elif algorithm == "rf":
                model_instance = RandomForestRegressor(max_depth=4, n_estimators=100)
            elif algorithm == "svr":
                model_instance = SVR()

            model_instance = model_instance.fit(train_x, train_y)
            y_hats = model_instance.predict(test_x)

        r2 = r2_score(test_y, y_hats)
        rmse = math.sqrt(mean_squared_error(test_y, y_hats, squared=False))
        return max(r2,0), rmse