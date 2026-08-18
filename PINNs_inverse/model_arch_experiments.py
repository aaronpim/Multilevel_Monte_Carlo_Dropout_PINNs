import os
import torch
import pandas as pd
from src.trainmodel import train_model

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "epochs": 3000,
        "num_modes": 1,
        "num_lay": 2,
        "hid_dim": 64,
        "smoothing_coef": 0.0,
        "drop_p": 0.05,
        "num_x_points": 3000,
        "num_y_points": 3000,
        "num_drop_evals": 20,
        "clamp": 1e-12,
        "lr": 1e-3
        }
    CONFIG.update(overwrites)
    return CONFIG

def model_vs_hid_dim( hid_dim_vec = [16, 32, 48, 64, 80, 96, 112, 128], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for hd in hid_dim_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, hid_dim = hd)
            final_lss, final_err, _, _ = train_model(CONFIG)
            output_vec.append([seed, hd, final_lss.item(), final_err.item()])
            print("")
            print(f"Completed seed {seed} and hidden dimension {hd}")
            print("")
    log_path = base_dir +'/model_vs_hid_dim.csv'
    pd.DataFrame(output_vec, columns=["seed", "hidden dimension", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_modes( modes_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for m in modes_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, num_modes = m)
            final_lss, final_err, _, _ = train_model(CONFIG)
            output_vec.append([seed, m, final_lss.item(), final_err.item()])
            print("")
            print(f"Completed seed {seed} and modes {m}")
            print("")
    log_path = base_dir +'/model_vs_num_modes.csv'
    pd.DataFrame(output_vec, columns=["seed", "modes", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_x_points( x_points_vec = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for X in x_points_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, num_x_points = X)
            final_lss, final_err, _, _ = train_model(CONFIG)
            output_vec.append([seed, X, final_lss.item(), final_err.item()])
            print("")
            print(f"Completed seed {seed} and number of x points {X}")
            print("")
    log_path = base_dir +'/model_vs_x_points.csv'
    pd.DataFrame(output_vec, columns=["seed", "N_x", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_y_points( y_points_vec = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for Y in y_points_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, num_y_points = Y)
            final_lss, final_err, _, _ = train_model(CONFIG)
            output_vec.append([seed, Y, final_lss.item(), final_err.item()])
            print("")
            print(f"Completed seed {seed} and number of y points {Y}")
            print("")
    log_path = base_dir +'/model_vs_y_points.csv'
    pd.DataFrame(output_vec, columns=["seed", "N_y", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_drop_p( drop_p_vec = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for p_drop in drop_p_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, drop_p = p_drop)
            final_lss, final_err, _, _ = train_model(CONFIG)
            output_vec.append([seed, p_drop, final_lss.item(), final_err.item()])
            print("")
            print(f"Completed seed {seed} and dropout prob {p_drop}")
            print("")
    log_path = base_dir +'/model_vs_drop_p.csv'
    pd.DataFrame(output_vec, columns=["seed", "dropout probability", "loss", "error"]).to_csv(log_path, index=False)

if __name__ == "__main__":
    model_vs_hid_dim()
    model_vs_modes()
    model_vs_x_points()
    model_vs_y_points()
    model_vs_drop_p()
