from si_module import SIModule
import torch
import torch.nn as nn


class MNDI(SIModule):
    def __init__(self, device):
        super().__init__(device, 2)
        self.alpha = nn.Parameter(torch.rand(1))

    def forward(self, outs):
        r_is = outs[0]
        r_js = outs[1]
        r_ks = outs[2]
        diff = r_js - self.alpha*r_ks
        up = r_is - diff
        down = r_is + diff
        mndis = up/down
        return mndis