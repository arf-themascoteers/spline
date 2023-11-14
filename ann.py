import torch.nn as nn
import torch
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sis = [
            {"si":BI, "count":5},
            {"si":DI, "count":0},
            {"si":RI, "count":0},
            {"si":NDI, "count":0},
            {"si":SNDI, "count":0},
            {"si":MNDI, "count":0}
        ]

        self.total = sum([si["count"] for si in self.sis])

        self.linear1 = nn.Sequential(
            nn.Linear(self.total, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )

        self.indices = torch.linspace(0, 1, 66).to(self.device)
        modules = []
        for si in self.sis:
            modules = modules + [si["si"]() for i in range(si["count"])]
        self.machines = nn.ModuleList(modules)

    def forward(self, x):
        outputs = torch.zeros(x.shape[0], self.total, dtype=torch.float32).to(self.device)
        x = x.permute(1,0)
        coeffs = natural_cubic_spline_coeffs(self.indices, x)
        spline = NaturalCubicSpline(coeffs)

        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)

        soc_hat = self.linear1(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_params(self):
        params = []
        index = 0
        for type_count, si in enumerate(self.sis):
            for i in range(si["count"]):
                machine = self.machines[index]
                p = {}
                p["si"] = si["si"].__name__
                p["params"] = machine.param_values()
                params.append(p)
                index = index+1
        return params
