import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

df = pd.read_csv("data/dataset_r.csv")
start_col = list(df.columns).index("400")
df2 = df.iloc[:,0:start_col]
nrows = len(df2)
for i in range(66):
    df2.insert(len(df2.columns), f"{i}", pd.Series([0]*nrows))
df2.to_csv("data/dataset_66.csv", index=False)