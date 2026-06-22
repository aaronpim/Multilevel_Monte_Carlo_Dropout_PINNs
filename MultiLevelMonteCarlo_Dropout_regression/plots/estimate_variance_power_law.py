import pandas as pd
import numpy as np

def power_law_loss(value = "hidden_dimension", filename = 'model_vs_hid_dim.csv'):
    df = pd.read_csv(filename)
    # Ensure x values are sorted
    x_values = sorted(df[value].unique())

    try:
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

        train_slope, train_intercept = np.polyfit(np.log(x_values), np.log(train_means), 1)
        hold_slope, hold_intercept = np.polyfit(np.log(x_values), np.log(hold_means), 1)

        print(f"--- Power-Law Decay Estimation ---")
        print(f"Train Loss Decay Rate (alpha): {-train_slope:.4f}")
        print(f"Hold Loss Decay Rate (alpha):  {-hold_slope:.4f}")
        print("----------------------------------")

    except:
        alea_data = [
            df[df[value] == x]["aleatoric variance error"].values
            for x in x_values
        ]

        total_data = [
            df[df[value] == x]["variance error"].values
            for x in x_values
        ]

        # Means
        alea_means = [np.mean(v) for v in alea_data]
        total_means  = [np.mean(v) for v in total_data]

        alea_slope, _ = np.polyfit(np.log(x_values), np.log(alea_means), 1)
        total_slope, _ = np.polyfit(np.log(x_values), np.log(total_means), 1)

        print(f"--- Power-Law Decay Estimation ---")
        print(f"Parametric Decay Rate: {-alea_slope:.4f}")
        print(f"Total Variance Decay Rate:  {-total_slope:.4f}")
        print("----------------------------------")

if __name__ == "__main__":
    power_law_loss(value = "hidden_dimension", filename = 'model_vs_hid_dim.csv')
