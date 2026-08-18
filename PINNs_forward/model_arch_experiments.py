import os
import torch
import pandas as pd
from src.model_defn import load_PirateNet
from src.trainmodel import train_model, CONFIG_to_folder_path
from src.loss_defn import pinns_loss, bcs_loss, estimate_error

N_total = 11
N_dropout = 100

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "input_dim": 2,
        "hidden_dim": 32,
        "num_blocks": 4,
        "output_dim": 1,
        "p_drop": 0.05,
        "activation": "nn.SiLU()",
        "sigma": 1.0,
        "x_num": 101,
        "eps_num": 101,
        "eps_min": -8,
        "eps_max": 4,
        "BC_coef": 1.0,
        "lr": 1e-3,
        "epochs": 5000,
        }
    CONFIG.update(overwrites)
    return CONFIG

def model_vs_hid_dim( hid_dim_vec = [16, 32, 48, 64, 96, 128], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for hd in hid_dim_vec:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, hidden_dim = hd)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only = True))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            print([seed, hd, final_loss.item(), final_error.item()])
            output_vec.append([seed, hd, final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and hidden dimension {hd}")
            print("")
    log_path = base_dir +'/model_vs_hid_dim.csv'
    pd.DataFrame(output_vec, columns=["seed", "hidden dimension", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_drop( dropout_prob_vec = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for dp in dropout_prob_vec:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, p_drop = dp)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            output_vec.append([seed, dp, final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and dropout {dp}")
            print("")
    log_path = base_dir +'/model_vs_drop.csv'
    pd.DataFrame(output_vec, columns=["seed", "dropout probability", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_blocks(num_blocks_vec = [1, 2, 3, 4, 5, 6, 7, 8], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for nb in num_blocks_vec:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, num_blocks = nb)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            output_vec.append([seed, nb, final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and number of blocks {nb}")
            print("")
    log_path = base_dir +'/model_vs_blocks.csv'
    pd.DataFrame(output_vec, columns=["seed", "blocks", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_activation( activation_list = ["nn.ReLU()", "nn.SiLU()", "nn.GELU()", "nn.Tanh()", "nn.LeakyReLU()", "nn.Sigmoid()"], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for act in activation_list:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, activation = act)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            output_vec.append([seed, act[3:-2], final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and activation {act}")
            print("")
    log_path = base_dir +'/model_vs_activation.csv'
    pd.DataFrame(output_vec, columns=["seed", "activation function", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_x_num( x_num_vec = [51, 101, 151, 201, 251, 301], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for xnum in x_num_vec:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, x_num = xnum)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only = True))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            print([seed, xnum, final_loss.item(), final_error.item()])
            output_vec.append([seed, xnum, final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and number of x poinsts {xnum}")
            print("")
    log_path = base_dir +'/model_vs_x_num.csv'
    pd.DataFrame(output_vec, columns=["seed", "number of x points", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_eps_num( eps_num_vec = [51, 101, 151, 201, 251, 301], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for en in eps_num_vec:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, eps_num = en)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only = True))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            print([seed, en, final_loss.item(), final_error.item()])
            output_vec.append([seed, en, final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and number of epsilon poinsts {en}")
            print("")
    log_path = base_dir +'/model_vs_eps_num.csv'
    pd.DataFrame(output_vec, columns=["seed", "number of epsilon points", "loss", "error"]).to_csv(log_path, index=False)

def model_vs_pts_num( pts_num_vec = [51, 101, 151, 201, 251], base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for pts in pts_num_vec:
        for seed in range(N_total):
            CONFIG = make_config(seed = seed, eps_num = pts, x_num = pts)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only = True))
                model.train()
                final_loss = 0
                final_error= 0
                for _ in range(N_dropout):
                    final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
                    final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
            else:
                final_loss, final_error = train_model(CONFIG)
            print([seed, pts, final_loss.item(), final_error.item()])
            output_vec.append([seed, pts, final_loss.item(), final_error.item()])
            print("")
            print(f"Completed seed {seed} and number of points {pts}")
            print("")
    log_path = base_dir +'/model_vs_pts_num.csv'
    pd.DataFrame(output_vec, columns=["seed", "number of points", "loss", "error"]).to_csv(log_path, index=False)


if __name__ == "__main__":
    # model_vs_hid_dim()
    # model_vs_drop()
    # model_vs_blocks()
    # model_vs_activation()
    # model_vs_x_num()
    # model_vs_eps_num()
    model_vs_pts_num()
