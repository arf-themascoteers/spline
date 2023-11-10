from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

df = pd.read_csv("data/dataset_66.csv")
signal = df.iloc[0].to_numpy()
start_index = list(df.columns).index("0")
signal = signal[start_index:]

# print(signal)
# print(signal.shape)


indices = np.arange(len(signal))
cubic_spline = CubicSpline(indices, signal)
fine_x = np.linspace(0, len(signal) - 1, 100)
#print(fine_x)
y_interp = cubic_spline([32.000001])
print(y_interp)
# print(y_interp.shape)

