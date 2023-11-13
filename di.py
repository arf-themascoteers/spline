from si_module import SIModule


class DI(SIModule):
    def __init__(self, device):
        super().__init__(device, 2)

    def forward(self, outs):
        return outs[0] - outs[1]