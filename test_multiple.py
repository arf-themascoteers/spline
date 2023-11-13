from evaluator import Evaluator
import spec_utils
from clear import clear_all

if __name__ == "__main__":
    clear_all()
    c = Evaluator(prefix="test_multiple", folds=10, repeat=3, algorithm="ann_multiple", column_group=spec_utils.get_wavelengths())
    c.process()