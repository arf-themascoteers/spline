from si_module import SIModule


class BI(SIModule):
    def __init__(self):
        super().__init__(1)

    def _forward(self, outs):
        return outs[0]

    def _names(self):
        return ["band"]