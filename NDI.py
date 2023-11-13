import torch.nn as nn
import torch
import torch.nn.functional as F
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANNSimple(nn.Module):
    def __init__(self, device, input_size, X_columns, y_column):
        super().__init__()
        self.device = device

        self.i = nn.Parameter(torch.tensor(0.1))
        self.j = nn.Parameter(torch.tensor(0.9))
        self.indices = torch.linspace(0, 1, 66).to(device)

    def forward(self, x, soc):
        x = x.permute(1,0)
        coeffs = natural_cubic_spline_coeffs(self.indices, x)
        spline = NaturalCubicSpline(coeffs)
        r_is = spline.evaluate(self.i)
        r_js = spline.evaluate(self.j)
        ndis = (r_is - r_js) / (r_is + r_js)
        return ndis.reshape(-1,1)