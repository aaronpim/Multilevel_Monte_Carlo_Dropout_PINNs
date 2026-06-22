import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def model_plot(name = "model_vs_activation", value = "activation_function", xlabel = 'activation function'):
    # Load CSV
    df = pd.read_csv(name+'.csv')

    # Ensure x values are sorted
    x_values = sorted(df[value].unique())

    # Prepare data
    train_data = [
        df[df[value] == x]["train_loss"].values
        for x in x_values
    ]

    hold_data = [
        df[df[value] == x]["hold_loss"].values
        for x in x_values
    ]

    # Means
    train_means = [np.mean(v) for v in train_data]
    hold_means  = [np.mean(v) for v in hold_data]

    # Base x positions
    x = np.arange(len(x_values))

    # Offset for side-by-side boxplots
    offset = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("log")

    bp1 = ax.boxplot(
        train_data,
        positions=x,
        widths=0.35,
        patch_artist=True,
    )
    for box in bp1["boxes"]:
        box.set(facecolor="orange", alpha=0.6)

    bp2 = ax.boxplot(
        hold_data,
        positions=x + offset,
        widths=0.35,
        patch_artist=True,
    )
    for box in bp2["boxes"]:
        box.set(facecolor="blue", alpha=0.6)

    if value != "activation_function":
        ax.plot(
            x,
            train_means,
            color="darkorange",
            marker="o",
            linewidth=2,
            label="mean training loss",
        )

        ax.plot(
            x,
            hold_means,
            color="darkblue",
            marker="o",
            linewidth=2,
            label="mean holdout loss",
        )

        ax.legend()
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
    model_plot('model_vs_hid_dim', 'hidden_dimension', 'hidden dimension')

    model_plot('model_vs_depth', 'num_hid_layers', 'num hidden layers')

    model_plot('model_vs_drop', 'dropout_prob', 'dropout probability')
