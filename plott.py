import matplotlib
matplotlib.use("TkAgg")
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def init():
    for ax in axes.flatten():
        ax.plot(x, np.ma.array(x, mask=True))
    return lines


def animate(frame):
    y_values = [np.sin(2 * np.pi * (x - 0.02 * frame + i)) for i in range(16)]

    for i, line in enumerate(lines):
        line.set_ydata(y_values[i])

    return lines


if __name__ == "__main__":
    df = pd.read_csv("test2.csv")
    epoch_col = df.columns.get_loc("epoch")
    batch_col = df.columns.get_loc("batch")
    rw_col = df.columns.get_loc("r2")
    rmse_col = df.columns.get_loc("rmse")
    itrs = len(df)
    sis = [{"name":"r2"},{"name":"rmse"}]
    si = None
    for index in range(rmse_col+1,len(df.columns)):
        col = df.columns[index]
        if "#" in col:
            if si is None:
                si = {}
            else:
                sis.append(si)
                si = {}
            si["name"] = col
            si["display_name"] = df.iloc[0,index]
            si["params"] = []
        else:
            si["params"].append(col)
    sis.append(si)
    print(sis)
    total_plots = len(sis)
    rows = math.ceil(total_plots/2)

    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 8))

    axes = axes.flatten()

    for i,p in enumerate(sis):
        name = p["name"]
        ax = axes[i]

        if name in ["r2", "rmse"]:
            data = df[name].tolist()
            ax.plot(data)
        else:
            for a_param in p["params"]:
                data= df[name].tolist()
                ax.plot(data)

    plt.show()
    # for ax in axes:
    #     ax.legend()
    #
    # ani = FuncAnimation(fig, animate, frames=np.arange(1, 200), init_func=init, blit=True)
    # plt.show()
    # ani.save('animation.mp4', writer='ffmpeg', fps=30)

