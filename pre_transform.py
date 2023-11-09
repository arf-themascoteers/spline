import pandas as pd
import numpy as np

df = pd.read_csv("data/dataset.csv")
start_col = list(df.columns).index("400")
df.iloc[:,start_col:] = np.round(1/(10**df.iloc[:,start_col:]),5)
df.to_csv("data/dataset_r.csv", index=False)

