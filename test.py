from evaluator import Evaluator
import spec_utils

if __name__ == "__main__":
    column_groups = [
        spec_utils.get_wavelengths(),
        spec_utils.get_wavelengths() + ["savi"]
    ]
    algorithms = ["ann_simple"]
    c = Evaluator(prefix="test", folds=10, algorithms=algorithms, column_groups=column_groups)
    c.process()