import pandas as pd
import pywt
import numpy as np

df = pd.read_csv("data/dataset_r.csv")
start_col = list(df.columns).index("400")
df2 = df.iloc[:,0:start_col]
nrows = len(df2)
for i in range(66):
    df2.insert(len(df2.columns), f"{i}", pd.Series([0.0]*nrows))


start_col = list(df2.columns).index("0")
for i in range(nrows):
    signal = df.iloc[i, start_col:]
    signal, _, _, _, _, _, _ = pywt.wavedec(signal, 'db1', level=6)
    df2.iloc[i, start_col:] = np.round(signal,5)


df2.to_csv("data/dataset_66.csv", index=False)