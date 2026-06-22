import os
import time
import torch
import pandas as pd
import numpy as np
from src.model_defn import RegressionModel, load_model
from scipy.spatial import Delaunay
import json
from math import factorial

def set_seed(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(device = 'cuda' if torch.cuda.is_available() else 'cpu', holdout = 6, normalise = True, log = True, eps = 1e-12):

    data = torch.load("Data/data_3.pt", weights_only = True).to(device).float()
    # Cheat to save memory
    # x = torch.load("Data/X.pt", weights_only = True).to(device).float()
    # y = torch.load("Data/Y.pt", weights_only = True).to(device).float()
    # z = torch.load("Data/Z.pt", weights_only = True).to(device).float()
    # dx = x[0,1,0] - x[0,0,0]
    # dy = y[1,0,0] - y[0,0,0]
    # dz = z[0,0,1] - z[0,0,0]
    integration_factor = torch.tensor([(2**3)/(3**3)]).to(device).float()
    if log is True:
        data = torch.log(data + eps)

    if normalise is True:
        factors = [integration_factor, data.min(), data.max()]
        data = (data - factors[0])/(factors[1] - factors[0])
    else:
        factors = [integration_factor]

    hold_data = data[-holdout:,:,:,:]
    data = data[:-holdout,:,:,:]

    values = torch.tensor(pd.read_csv("Data/generated_values.csv").values).to(device).float()
    hold_values = values[-holdout:,:]
    values = values[:-holdout,:]


    return values, data.permute([1,0,2,3,4]), hold_values, hold_data.permute([1,0,2,3,4]), factors

def eval_model(model, values, reps = 10):
    return torch.stack([model(values) for _ in range(reps)], dim = 0)

def compute_delaunay_weights(values,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    values_np = values.cpu().numpy()
    tri = Delaunay(values_np)
    n_points = values_np.shape[0]
    dim = values_np.shape[1]
    w = np.zeros(n_points)
    for simplex in tri.simplices:
        pts = values_np[simplex]
        base = pts[0]
        M = pts[1:] - base
        volume = abs(np.linalg.det(M)) / factorial(dim)
        w[simplex] += volume / (dim + 1)
    return torch.tensor(w, device=device)

def loss(model, values, data, factors, power = 2, weights = None):
    if weights is None:
        weights = compute_delaunay_weights(values, device = values.device)
    output = eval_model(model, values, reps = data.shape[0])
    res = torch.abs(output - data)**power
    integrand = factors[0]*torch.sum(res, dim = [2,3,4])
    weighted_sum = (integrand * weights).sum(dim=-1)
    output = weighted_sum.mean()
    return output**(1/power)

def load_all(CONFIG, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    values, data, hold_values, hold_data, factors = load_data(device = device, holdout = CONFIG["holdout"], normalise = CONFIG["normalise"], log = CONFIG["log"], eps = CONFIG["eps"])
    model = load_model(CONFIG,  input_dim = values.shape[1], output_dims=[data.shape[2], data.shape[3], data.shape[4]], device = device)
    weights = compute_delaunay_weights(values, device = values.device)
    hold_weights = compute_delaunay_weights(hold_values, device = hold_values.device)
    return model, values, data, hold_values, hold_data, factors, weights, hold_weights

def CONFIG_to_folder_path(CONFIG, base_dir = 'runs'):
    os.makedirs(base_dir, exist_ok=True)
    param_str = "_".join(f"{k}-{v}" for k, v in CONFIG.items())
    folder_name = f"{param_str}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def ETA_calculation(epoch, maxepochs, start_time, loss, hold_loss, best_loss):
    elapsed = time.time() - start_time
    avg_time_per_epoch = elapsed/(epoch+1)
    eta_seconds = avg_time_per_epoch * (maxepochs - epoch)
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    print(f"\rEpoch {epoch+1}/{maxepochs} | ETA: {eta_str} | Loss: {loss:.5e} | Holdout loss: {hold_loss:.5e} | Best Loss: {best_loss:.5e}", end="", flush=True)

def train_model(CONFIG, device = 'cuda' if torch.cuda.is_available() else 'cpu', base_dir = 'runs'):
    print(device)
    folder_path = CONFIG_to_folder_path(CONFIG, base_dir = base_dir)
    set_seed(CONFIG["seed"])
    model, values, data, hold_values, hold_data, factors, weights, hold_weights = load_all(CONFIG, device = device)
    optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    history    = []
    start_time = time.time()
    best_loss  = float('inf')
    for i in range(CONFIG["epochs"]):
        model.train()
        optimiser.zero_grad()
        regloss = loss(model, values, data, factors, power = CONFIG["power"], weights = weights)
        regloss.backward()
        optimiser.step()
        with torch.no_grad():
            hold_loss = loss(model, hold_values, hold_data, factors, power = CONFIG["power"], weights = hold_weights)
            history.append([regloss.item(), hold_loss.item()])
            if hold_loss.item() < best_loss:
                best_loss = hold_loss.item()
                best_dict = model.state_dict()
            ETA_calculation(i, CONFIG["epochs"], start_time, regloss.item(), hold_loss.item(), best_loss)
    model.load_state_dict(best_dict)
    save_path = os.path.join(folder_path, 'model.pt')
    torch.save(model.state_dict(), save_path)
    log_path = os.path.join(folder_path, 'loss.csv')
    pd.DataFrame(history, columns=["train_loss", "hold_loss"]).to_csv(log_path, index=False)
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    with torch.no_grad():
        final_loss = loss(model, values, data, factors, power = CONFIG["power"], weights = weights)
        final_hold_loss = loss(model, hold_values, hold_data, factors, power = CONFIG["power"], weights = hold_weights)
    return final_loss, final_hold_loss
    print("")
if __name__ == "__main__":
    CONFIG = {
        "seed": 0,
        "holdout": 6,
        "normalise": 1,
        "log": 1,
        "eps": 1e-12,
        "num_hid_layers": 3,
        "hid_dim": 256,
        "activation": "nn.ReLU()",
        "dropout_prob": 0.05,
        "epochs": 2000,
        "lr": 1e-4,
        "power": 2,
        }
    model, values, data, hold_values, hold_data, factors, weights, hold_weights = load_all(CONFIG, device = 'cpu')
    print(loss(model, values, data, factors, power = 2, weights = weights) )
