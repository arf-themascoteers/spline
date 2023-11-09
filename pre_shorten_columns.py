import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

df = pd.read_csv("data/dataset_r.csv")

signal = df.iloc[0].to_numpy()

start_index = list(df.columns).index("400")

signal = signal[start_index:]



(cA, cD) = pywt.dwt(signal, 'db1')
(cA, cD) = pywt.dwt(cA, 'db1')
(cA, cD) = pywt.dwt(cA, 'db1')
(cA, cD) = pywt.dwt(cA, 'db1')
(cA, cD) = pywt.dwt(cA, 'db1')
(cA, cD) = pywt.dwt(cA, 'db1')
print(len(cA))

signal = cA
x = np.arange(len(signal))

plt.figure()
plt.plot(x, signal, 'o', label='Bands', markersize=2)
plt.xlabel('i')
plt.ylabel('b(i) ')
#plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()

signal = cA
x = np.arange(len(signal))  # Creating x values based on the array index
print(x)
cs = CubicSpline(x, signal)
new_x = np.linspace(0, len(signal) - 1, 100)  # 100 points covering the range

# Interpolate the new y values
new_y = cs(new_x)

# Plot the original data points and the cubic spline interpolation
plt.figure()
plt.plot(x, signal, 'o', label='Bands',markersize=2)
plt.plot(new_x, new_y, label='Cubic Spline Interpolation',markersize=0.25)
plt.xlabel('i')
plt.ylabel('b(i) ')
plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()