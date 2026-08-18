import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def model_plot(name="model_vs_drop",
               value="dropout probability",
               xlabel="dropout probability"):

    # Load CSV
    df = pd.read_csv(name + ".csv")

    # Ensure x values are sorted
    x_values = sorted(df[value].unique())

    # Prepare data
    loss_data = [
        df[df[value] == x]["loss"].values
        for x in x_values
    ]

    error_data = [
        df[df[value] == x]["error"].values
        for x in x_values
    ]

    # Means
    loss_means  = [np.mean(v) for v in loss_data]
    error_means = [np.mean(v) for v in error_data]

    x = np.arange(len(x_values))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Optional: use log scale on both
    #ax1.set_yscale("log")
    #ax2.set_yscale("log")

    # Slight offset so the boxplots don't overlap
    width = 0.3
    offset = width / 2

    # -----------------
    # Loss (left axis)
    # -----------------
    bp1 = ax1.boxplot(
        loss_data,
        positions=x - offset,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
    )

    for box in bp1["boxes"]:
        box.set(facecolor="orange", alpha=0.6)

    line1, = ax1.plot(
        x - offset,
        loss_means,
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Mean loss",
    )

    ax1.set_ylabel("Loss", color="darkorange")
    ax1.tick_params(axis="y", colors="darkorange")
    ax1.spines["left"].set_color("darkorange")

    # -----------------
    # Error (right axis)
    # -----------------
    bp2 = ax2.boxplot(
        error_data,
        positions=x + offset,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
    )

    for box in bp2["boxes"]:
        box.set(facecolor="cornflowerblue", alpha=0.6)

    line2, = ax2.plot(
        x + offset,
        error_means,
        color="darkblue",
        marker="o",
        linewidth=2,
        label="Mean error",
    )

    ax2.set_ylabel("Error", color="darkblue")
    ax2.tick_params(axis="y", colors="darkblue")
    ax2.spines["right"].set_color("darkblue")

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_values)
    ax1.set_xlabel(xlabel)

    # Combined legend
    #ax1.legend(handles=[line1, line2], loc="best")

    plt.tight_layout()
    plt.savefig(name + ".pdf")
    plt.close()

def power_law_loss(value = "hidden dimension", filename = 'model_vs_hid_dim.csv'):
    df = pd.read_csv(filename)
    # Ensure x values are sorted
    x_values = sorted(df[value].unique())

    try:
        # Prepare data
        train_data = [
            df[df[value] == x]["loss"].values
            for x in x_values
        ]

        hold_data = [
            df[df[value] == x]["error"].values
            for x in x_values
        ]

        # Means
        train_means = [np.mean(v) for v in train_data]
        hold_means  = [np.mean(v) for v in hold_data]

        #train_slope, train_intercept = np.polyfit(np.log(x_values), np.log(train_means), 1)
        #hold_slope, hold_intercept = np.polyfit(np.log(x_values), np.log(hold_means), 1)
        train_slope, train_intercept = np.polyfit(np.array(x_values), np.array(train_means), 1)
        hold_slope, hold_intercept = np.polyfit(np.array(x_values), np.array(hold_means), 1)
        print(f"--- Power-Law Decay Estimation ---")
        print(f"Loss Decay Rate (alpha): {-train_slope:.4e}")
        print(f"Error Decay Rate (alpha):  {-hold_slope:.4e}")
        print("----------------------------------")
    except:
        1+1
if __name__ == "__main__":
    # power_law_loss()
    # model_plot(
    #     name="model_vs_hid_dim",
    #     value="hidden dimension",
    #     xlabel="hidden dimension",
    # )
    # model_plot(
    #     name="model_vs_num_modes",
    #     value="modes",
    #     xlabel="number of modes",
    # )
    model_plot(
        name="model_vs_x_points",
        value="N_x",
        xlabel=rf"$N_x$",
    )
    # model_plot(
    #     name="model_vs_y_points",
    #     value="N_y",
    #     xlabel=rf"$N_y$",
    # )
    # power_law_loss(value = "N_y", filename = 'model_vs_y_points.csv')
    # model_plot(
    #     name="model_vs_drop_p",
    #     value="dropout probability",
    #     xlabel="dropout probability",
    # )
