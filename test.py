from evaluator import Evaluator
from plott import plotit

if __name__ == "__main__":
    c = Evaluator()
    r2, rmse = c.process()
    print("r2",round(r2,5))
    print("rmse",round(rmse,5))
    plotit()