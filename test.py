from evaluator import Evaluator
import spec_utils
from clear import clear_all

if __name__ == "__main__":
    clear_all()
    c = Evaluator(prefix="test")
    c.process()