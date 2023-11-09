import pandas as pd
import pywt
import matplotlib.pyplot as plt

df = pd.read_csv("data/dataset_min.csv")
signal = df.iloc[0].to_numpy()
start_index = list(df.columns).index("400")
signal = signal[start_index:]
plt.plot(signal)
plt.show()

signal,_,_,_,_,_,_ = pywt.wavedec(signal, 'db1', level=6)
print(signal.shape)
plt.plot(signal)
plt.show()

