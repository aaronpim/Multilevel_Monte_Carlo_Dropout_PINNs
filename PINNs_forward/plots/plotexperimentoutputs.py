import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def model_plot(name = "model_vs_drop", value = "dropout probability", xlabel = 'dropout probability'):
    # Load CSV
    df = pd.read_csv(name+'.csv')

    # Ensure x values are sorted
    x_values = sorted(df[value].unique())

    # Prepare data
    loss = [
        df[df[value] == x]["loss"].values
        for x in x_values
    ]

    error = [
        df[df[value] == x]["error"].values
        for x in x_values
    ]

    # Means
    loss_means = [np.exp(np.mean(np.log(v))) for v in loss]
    error_means  = [np.exp(np.mean(np.log(v))) for v in error]

    # Base x positions
    x = np.arange(len(x_values))

    # Offset for side-by-side boxplots
    offset = 0.1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("log")

    bp1 = ax.boxplot(
        loss,
        positions=x,
        widths=0.35,
        patch_artist=True,
    )
    for box in bp1["boxes"]:
        box.set(facecolor="orange", alpha=0.6)

    bp2 = ax.boxplot(
        error,
        positions=x + offset,
        widths=0.35,
        patch_artist=True,
    )
    for box in bp2["boxes"]:
        box.set(facecolor="blue", alpha=0.6)

    for flier in bp1["fliers"]:
        flier.set(marker='o', markeredgecolor='darkorange', markersize=4, alpha=0.6,)
    for flier in bp2["fliers"]:
        flier.set(marker='o', markeredgecolor='navy', markersize=4, alpha=0.6,)


    if not name == "model_vs_activation":
        ax.plot(
            x,
            loss_means,
            color="darkorange",
            marker="o",
            linewidth=2,
            label="mean loss",
        )

        ax.plot(
            x+ offset,
            error_means,
            color="darkblue",
            marker="o",
            linewidth=2,
            label="mean error",
        )

        ax.legend()
        P_loss  = np.polyfit( np.log(x_values), np.log(loss_means), 1)
        P_error = np.polyfit( np.log(x_values), np.log(error_means),1)
        print(P_loss[0], 'rate of decay for the loss of', name)
        print(P_error[0], 'rate of decay for the error of', name)

        with open("decay_rates.txt", "a") as f:
            f.write(f"{P_loss[0]} rate of decay for the loss of {name}\n")
            f.write(f"{P_error[0]} rate of decay for the error of {name}\n")
    else:
        legend_handles = [ Patch(facecolor="orange", edgecolor="darkorange", alpha=0.5, label="Loss"), Patch(facecolor="blue", edgecolor="navy", alpha=0.5, label="Error"),]
        ax.legend(handles=legend_handles, loc="best")
    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss")
    plt.tight_layout()
    plt.savefig(name+'.pdf')
    plt.close()


if __name__ == "__main__":
    model_plot(name = "model_vs_hid_dim", value = "hidden dimension", xlabel = 'hidden dimension')
    model_plot(name = "model_vs_drop", value = "dropout probability", xlabel = 'dropout probability')
    model_plot(name = "model_vs_blocks", value = "blocks", xlabel = 'blocks')
    model_plot(name = "model_vs_activation", value = "activation function", xlabel = 'activation function')
    model_plot(name = "model_vs_x_num", value = "number of x points", xlabel = 'number of x points')
    model_plot(name = "model_vs_eps_num", value = "number of epsilon points", xlabel = 'number of epsilon points')
    model_plot(name = "model_vs_pts_num", value = "number of points", xlabel = 'number of points')
