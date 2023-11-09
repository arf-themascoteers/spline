import pandas as pd

df = pd.read_csv("data/dataset_r.csv")
df = df.sample(frac=0.01)
df.to_csv("data/dataset_min.csv", index=False)