import torch.nn as nn
import torch
import torch.nn.functional as F


class SIModule(nn.Module):
    def __init__(self, device, count_params):
        super().__init__()
        self.device = device
        self.params = nn.Parameter(torch.rand(count_params))

    def forward(self, spline):
        outs = [spline.evaluate(F.sigmoid(param)) for param in self.params]
        outs = self._forward(outs)
        return outs

    def _forward(self, spline):
        pass

    def lower_bound_loss(self):
        loss = torch.zeros(1, dtype=torch.float32).to(self.device)
        for param in self.params:
            loss = loss + F.relu(-1*param)
        return loss

    def upper_bound_loss(self):
        loss = torch.zeros(1, dtype=torch.float32).to(self.device)
        for param in self.params:
            loss = loss + F.relu(param - 1)
        return loss
