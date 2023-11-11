from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

df = pd.read_csv("data/dataset_66.csv")
signal = df.iloc[0].to_numpy()
start_index = list(df.columns).index("0")
signal = signal[start_index:]

print("signal")
print(signal)
print(signal.shape)


indices = np.linspace(0, 1, 66)
cubic_spline = CubicSpline(indices, signal)


y_interp = cubic_spline([0])
print(y_interp)
y_interp = cubic_spline([0.5])
print(y_interp)
y_interp = cubic_spline([1])
print(y_interp)



