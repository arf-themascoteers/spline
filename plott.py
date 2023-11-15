import matplotlib
matplotlib.use("TkAgg")
import math
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("test.csv")
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
    rows = math.ceil(total_plots/4)

    fig, axes = plt.subplots(nrows=rows, ncols=4)

    axes = axes.flatten()

    for i,p in enumerate(sis):
        name = p["name"]
        ax = axes[i]
        title = False
        if name in ["r2", "rmse"]:
            data = df[name].tolist()
            ax.plot(data)
            ax.set_title(name)
        else:
            for a_param in p["params"]:
                data= df[a_param].tolist()
                ax.plot(data, label=a_param.split(".")[0])
                ax.set_ylim(1, 4300)
                if not title:
                    ax.set_title(df.loc[0,name])
        ax.legend()


    plt.show()

