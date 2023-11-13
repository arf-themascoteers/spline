import torch.nn as nn
import torch
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from mndi import MNDI
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion_soc = torch.nn.MSELoss(reduction='sum')

        self.count_bis = 10
        self.count_dis = 10
        self.count_ris = 0
        self.count_ndis = 10
        self.count_mndis = 10

        self.alpha = 1

        self.total = self.count_bis + self.count_dis + self.count_ris + self.count_ndis + self.count_mndis

        self.linear1 = nn.Sequential(
            nn.Linear(self.total, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )
        self.indices = torch.linspace(0, 1, 66).to(self.device)

        self.bis = nn.ModuleList([BI(self.device) for i in range(self.count_bis)])
        self.dis = nn.ModuleList([DI(self.device) for i in range(self.count_dis)])
        self.ris = nn.ModuleList([RI(self.device) for i in range(self.count_ris)])
        self.ndis = nn.ModuleList([NDI(self.device) for i in range(self.count_ndis)])
        self.mndis = nn.ModuleList([MNDI(self.device) for i in range(self.count_mndis)])

    def forward(self, x, soc):
        outputs_bis = torch.zeros(x.shape[0], self.count_bis, dtype=torch.float32).to(self.device)
        outputs_dis = torch.zeros(x.shape[0], self.count_dis, dtype=torch.float32).to(self.device)
        outputs_ris = torch.zeros(x.shape[0], self.count_ris, dtype=torch.float32).to(self.device)
        outputs_ndis = torch.zeros(x.shape[0], self.count_ndis, dtype=torch.float32).to(self.device)
        outputs_mndis = torch.zeros(x.shape[0], self.count_mndis, dtype=torch.float32).to(self.device)

        loss_bis = torch.zeros(self.count_bis, dtype=torch.float32).to(self.device)
        loss_dis = torch.zeros(self.count_dis, dtype=torch.float32).to(self.device)
        loss_ris = torch.zeros(self.count_ris, dtype=torch.float32).to(self.device)
        loss_ndis = torch.zeros(self.count_ndis, dtype=torch.float32).to(self.device)
        loss_mndis = torch.zeros(self.count_mndis, dtype=torch.float32).to(self.device)

        x = x.permute(1,0)
        coeffs = natural_cubic_spline_coeffs(self.indices, x)
        spline = NaturalCubicSpline(coeffs)

        for i,bi in enumerate(self.bis):
            outputs_bis[:,i], loss_bis[i] = bi(spline)

        for i,di in enumerate(self.dis):
            outputs_dis[:,i], loss_dis[i] = di(spline)

        for i,ri in enumerate(self.ris):
            outputs_ris[:,i], loss_ris[i] = ri(spline)

        for i,ndi in enumerate(self.ndis):
            outputs_ndis[:,i], loss_ndis[i] = ndi(spline)

        for i,mndi in enumerate(self.mndis):
            outputs_mndis[:,i], loss_mndis[i] = mndi(spline)

        outputs = torch.hstack((outputs_bis, outputs_dis, outputs_ris, outputs_ndis, outputs_mndis))

        soc_hat = self.linear1(outputs)
        soc_hat = soc_hat.reshape(-1)
        loss = self.criterion_soc(soc_hat, soc)
        for l in loss_bis:
            loss = loss + self.alpha * l
        for l in loss_dis:
            loss = loss + self.alpha * l
        for l in loss_ris:
            loss = loss + self.alpha * l
        for l in loss_ndis:
            loss = loss + self.alpha * l
        for l in loss_mndis:
            loss = loss + self.alpha * l
        return soc_hat, outputs, loss

    def get_params(self):
        return round(self.i.item()*4200), round(self.j.item()*4200)