import torch
import pandas as pd
from torchcubicspline import(natural_cubic_spline_coeffs,
                             NaturalCubicSpline)

df = pd.read_csv("data/dataset_66.csv")
start_index = list(df.columns).index("0")
signal = df.iloc[0:3,start_index:]
signal = signal.to_numpy()
signal = torch.tensor(signal,dtype=torch.float32)
signal = signal.permute(1,0)
print("signal")
print(signal[0,:])
print(signal[32,:])
print(signal[65,:])


indices = torch.linspace(0, 1, 66)
coeffs  = natural_cubic_spline_coeffs(indices, signal)
spline = NaturalCubicSpline(coeffs)
point = torch.tensor([0,0.492307,1])
out = spline.evaluate(point)
print(out)


