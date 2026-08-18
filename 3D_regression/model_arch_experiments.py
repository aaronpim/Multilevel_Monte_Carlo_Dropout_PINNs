import os
import torch
import pandas as pd
from src.trainmodel import CONFIG_to_folder_path, load_all, train_model

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "holdout": 5,
        "normalise": 0,
        "log": 0,
        "eps": 1e-12,
        "num_hid_layers": 4,
        "hid_dim": 128,
        "activation": "nn.ReLU()",
        "dropout_prob": 0.05,
        "epochs": 10000,
        "lr": 1e-4,
        "power": 2,
        }
    CONFIG.update(overwrites)
    return CONFIG

def model_vs_hid_dim( hid_dim_vec = [16, 32, 48, 64, 80, 96, 112, 128], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for hd in hid_dim_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, hid_dim = hd)
            final_loss, final_hold_loss = train_model(CONFIG)
            output_vec.append([seed, hd, final_loss.item(), final_hold_loss.item()])
            print("")
            print(f"Completed seed {seed} and hidden dimension {hd}")
            print("")
    log_path = base_dir +'/model_vs_hid_dim.csv'
    pd.DataFrame(output_vec, columns=["seed", "hidden_dimension", "train_loss", "hold_loss"]).to_csv(log_path, index=False)

def model_vs_drop( dropout_prob_vec = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for dp in dropout_prob_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, dropout_prob = dp)
            final_loss, final_hold_loss = train_model(CONFIG)
            output_vec.append([seed, dp, final_loss.item(), final_hold_loss.item()])
            print("")
            print(f"Completed seed {seed} and dropout {dp}")
            print("")
    log_path = base_dir +'/model_vs_drop.csv'
    pd.DataFrame(output_vec, columns=["seed", "dropout_prob", "train_loss", "hold_loss"]).to_csv(log_path, index=False)

def model_vs_depth(num_hid_layers_vec = [1, 2, 3, 4, 5, 6], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for nhl in num_hid_layers_vec:
        for seed in range(9):
            CONFIG = make_config(seed = seed, num_hid_layers = nhl)
            final_loss, final_hold_loss = train_model(CONFIG)
            output_vec.append([seed, nhl, final_loss.item(), final_hold_loss.item()])
            print("")
            print(f"Completed seed {seed} and number of hidden layers {nhl}")
            print("")
    log_path = base_dir +'/model_vs_depth.csv'
    pd.DataFrame(output_vec, columns=["seed", "num_hid_layers", "train_loss", "hold_loss"]).to_csv(log_path, index=False)

def model_vs_activation( activation_list = ["nn.ReLU()", "nn.SiLU()", "nn.GELU()", "nn.Tanh()", "nn.LeakyReLU()", "nn.Sigmoid()"], base_dir = 'plots'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for act in activation_list:
        for seed in range(9):
            CONFIG = make_config(seed = seed, activation = act)
            final_loss, final_hold_loss = train_model(CONFIG)
            output_vec.append([seed, act[3:-2], final_loss.item(), final_hold_loss.item()])
            print("")
            print(f"Completed seed {seed} and activation {act}")
            print("")
    log_path = base_dir +'/model_vs_activation.csv'
    pd.DataFrame(output_vec, columns=["seed", "activation_function", "train_loss", "hold_loss"]).to_csv(log_path, index=False)

if __name__ == "__main__":
    model_vs_hid_dim()
    model_vs_drop()
    model_vs_depth()
    model_vs_activation()
