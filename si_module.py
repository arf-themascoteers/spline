import torch.nn as nn
import torch
import torch.nn.functional as F


class SIModule(nn.Module):
    def __init__(self, count_params):
        super().__init__()
        self.params = nn.Parameter(torch.rand(count_params))

    def forward(self, spline):
        outs = [spline.evaluate(F.sigmoid(param)) for param in self.params]
        outs = self._forward(outs)
        return outs

    def _forward(self, spline):
        pass

