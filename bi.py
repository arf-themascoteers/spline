from si_module import SIModule


class BI(SIModule):
    def __init__(self, device):
        super().__init__(device, 1)

    def _forward(self, outs):
        return outs[0]