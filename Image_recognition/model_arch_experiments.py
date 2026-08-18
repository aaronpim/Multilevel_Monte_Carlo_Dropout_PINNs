import os
import torch
import pandas as pd
from src.trainmodel import train_model, default_config

def make_config(**overwrites):
    CONFIG = default_config()
    CONFIG.update(overwrites)
    return CONFIG

def model_vs_drop( dropout_prob_vec = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1,], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, "model_vs_drop.csv")
    if os.path.exists(log_path):
        output_df = pd.read_csv(log_path)
    else:
        output_df = pd.DataFrame(columns=["seed", "dropout probability", "training loss", "holdout loss",])
    for dp in dropout_prob_vec:
        for seed in range(9):
            already_done = ( (output_df["seed"] == seed) & (output_df["dropout probability"] == dp) ).any()

            if already_done:
                print(f"Skipping seed={seed}, dropout={dp}")
                continue
            CONFIG = make_config(seed = seed, drop_p = dp)
            final_loss, final_hold_loss = train_model(CONFIG)
            new_row = pd.DataFrame([[seed, dp, final_loss, final_hold_loss]], columns=output_df.columns, )
            output_df = pd.concat([output_df, new_row], ignore_index=True)
            output_df = output_df.sort_values(by=["dropout probability", "seed"]).reset_index(drop=True)
            output_df.to_csv(log_path, index=False)
            print("")
            print(f"Completed seed {seed} and dropout {dp}")
            print("")

def model_vs_width(width_vec = [8, 16, 24, 32, 48, 64, 96, 128], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for width in width_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, width = width)
            final_loss, final_hold_loss = train_model(CONFIG)
            output_vec.append([seed, width, final_loss, final_hold_loss])
            print("")
            print(f"Completed seed {seed} and width {width}")
            print("")
    log_path = base_dir +'/model_vs_width.csv'
    pd.DataFrame(output_vec, columns=["seed", "width", "training loss", "holdout loss"]).to_csv(log_path, index=False)

if __name__ == "__main__":
    model_vs_drop()
    model_vs_width()
