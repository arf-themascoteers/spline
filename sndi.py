from si_module import SIModule
import torch.nn as nn
import torch


class SNDI(SIModule):
    def __init__(self):
        super().__init__(2)
        self.alpha = nn.Parameter(torch.rand(1))

    def _forward(self, outs):
        scaled_j = self.alpha*outs[1]
        return (outs[0] - scaled_j)/(outs[0] + scaled_j)