import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.curves = {}

    def plot(self,plot_items):
        plt.clf()
        for p in plot_items:
            key = p["name"]
            value = p["value"]
            if key not in self.curves:
                self.curves[key] = []
            self.curves[key].append(value)
            plt.plot(self.curves[key])
        plt.draw()
