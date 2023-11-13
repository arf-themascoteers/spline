from evaluator import Evaluator
from clear import clear_all

if __name__ == "__main__":
    clear_all()
    c = Evaluator(name="test.csv")
    c.process()