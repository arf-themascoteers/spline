from si_module import SIModule


class NDI(SIModule):
    def __init__(self):
        super().__init__(2)

    def _forward(self, outs):
        return (outs[0] - outs[1])/(outs[0] + outs[1])