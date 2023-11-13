from si_module import SIModule
import torch
import torch.nn as nn


class MNDI(SIModule):
    def __init__(self):
        super().__init__(3)
        self.alpha = nn.Parameter(torch.rand(1))

    def _forward(self, outs):
        r_is = outs[0]
        r_js = outs[1]
        r_ks = outs[2]
        diff = r_js - self.alpha*r_ks
        up = r_is - diff
        down = r_is + diff
        mndis = up/down
        return mndis

    def _names(self):
        return ["i","j","k","alpha"]

    def param_values(self):
        param_value = super().param_values()
        param_value.append({"name":"alpha","value":self.alpha.item()})
        return param_value