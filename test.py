from evaluator import Evaluator
import spec_utils
from clear import clear_all

if __name__ == "__main__":
    clear_all()
    c = Evaluator(prefix="test", folds=3, repeat=2, algorithm="ann_simple", column_group=spec_utils.get_wavelengths())
    c.process()