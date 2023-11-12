import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline


class ANNSimple(nn.Module):
    def __init__(self, device, input_size, X_columns, y_column):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.X_columns = X_columns
        self.y_column = y_column
        self.criterion_soc = torch.nn.MSELoss(reduction='sum')
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

        self.i = nn.Parameter(torch.tensor(0.1))
        self.j = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, soc):
        indices = torch.linspace(0, 1, 66)
        cubic_splines = [CubicSpline(indices, x[index]) for index in range(x.shape[0])]
        r_i = torch.zeros(x.shape[0], dtype=torch.float32)
        r_j = torch.zeros(x.shape[0], dtype=torch.float32)
        for index in range(x.shape[0]):
            cs = cubic_splines[index]
            r_i[index] = cs([self.i.item()])
            r_j[index] = cs([self.j.item()])
        ndis = self.ndi(r_i, r_j)
        soc_hat = self.linear1(ndis)
        soc_hat = soc_hat.reshape(-1)
        loss = self.criterion_soc(soc_hat, soc)
        return soc_hat, ndis, loss

    def lower_bound_loss(self, param):
        return F.relu(-1*param)

    def upper_bound_loss(self, param):
        return F.relu(param - 1)

    def ndi(self, r_i, r_j):
        return (r_i - r_j) / (r_i + r_j)
