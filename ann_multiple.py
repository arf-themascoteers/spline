import torch.nn as nn
import torch
import torch.nn.functional as F
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANNMultiple(nn.Module):
    def __init__(self, device, input_size, X_columns, y_column):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.X_columns = X_columns
        self.y_column = y_column
        self.criterion_soc = torch.nn.MSELoss(reduction='sum')
        self.linear1 = nn.Sequential(
            nn.Linear(100, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

        self.i = nn.Parameter(torch.tensor(0.1))
        self.j = nn.Parameter(torch.tensor(0.9))
        self.indices = torch.linspace(0, 1, 66).to(device)

    def forward(self, x, soc):
        x = x.permute(1,0)
        coeffs = natural_cubic_spline_coeffs(self.indices, x)
        spline = NaturalCubicSpline(coeffs)
        r_is = spline.evaluate(self.i)
        r_js = spline.evaluate(self.j)
        ndis = self.ndi(r_is, r_js)
        soc_hat = self.linear1(ndis)
        soc_hat = soc_hat.reshape(-1)
        loss = self.criterion_soc(soc_hat, soc)
        return soc_hat, ndis, loss

    def lower_bound_loss(self, param):
        return F.relu(-1*param)

    def upper_bound_loss(self, param):
        return F.relu(param - 1)

    def get_params(self):
        return round(self.i.item()*4200), round(self.j.item()*4200)