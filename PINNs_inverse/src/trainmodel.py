import os
import json
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model_defn import load_model
from src.loss_defn import loss, get_device

def default_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "epochs": 3000,
        "num_modes": 1,
        "num_lay": 2,
        "hid_dim": 128,
        "smoothing_coef": 0.0,
        "drop_p": 0.05,
        "num_x_points": 2000,
        "num_y_points": 10000,
        "num_drop_evals": 20,
        "clamp": 1e-12,
        "lr": 1e-3
        }
    CONFIG.update(overwrites)
    return CONFIG

def set_seed(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CONFIG_to_folder_path(CONFIG, base_dir = 'runs'):
    os.makedirs(base_dir, exist_ok=True)
    param_str = "_".join(f"{k}-{v}" for k, v in CONFIG.items())
    folder_name = f"{param_str}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def ETA_calculation(epoch, maxepochs, start_time, lss, err):
    elapsed = time.time() - start_time
    avg_time_per_epoch = elapsed/(epoch+1)
    eta_seconds = avg_time_per_epoch * (maxepochs - epoch - 1)
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    print(f"\rEpoch {epoch+1}/{maxepochs} | ETA: {eta_str} | Loss: {lss:.5e}| Error: {err:.5e}", end="", flush=True)

def exact_u(x):
    x1 = x[:,0]*torch.cos(x[:,1])
    x2 = x[:,0]*torch.sin(x[:,1])
    normalisation_c  = math.sqrt(24/math.pi)
    return x1*x2*(1 - x[:,0]**2)*normalisation_c/12

def exact_f(x):
    x1 = x[:,0]*torch.cos(x[:,1])
    x2 = x[:,0]*torch.sin(x[:,1])
    normalisation_c  = math.sqrt(24/math.pi)
    return x1*x2*normalisation_c

def sunflower_disk_points(n, device="cpu"):
    k = torch.arange(n, dtype=torch.float32, device=device)
    r = torch.sqrt(k / n)
    theta = k * math.pi * (3 - math.sqrt(5))
    return torch.stack((r, theta), dim=1)

def train_model(CONFIG = None, device = 'cuda' if torch.cuda.is_available() else 'cpu', base_dir = 'runs'):
    if CONFIG is None:
        CONFIG = default_config()
    folder_path = CONFIG_to_folder_path(CONFIG, base_dir = base_dir)
    set_seed(CONFIG["seed"])
    model = load_model(CONFIG, device = device)
    model.train()
    optimiser  = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=200, threshold=1e-5, threshold_mode="rel", cooldown=50, min_lr=1e-7)
    x = sunflower_disk_points(CONFIG["num_x_points"], device = get_device(model) )
    data = exact_u(x)
    f = exact_f(x)
    history    = []
    start_time = time.time()
    for i in range(CONFIG["epochs"]):
        optimiser.zero_grad()
        lss = 0
        for _ in range(CONFIG["num_drop_evals"]):
            lss += loss(model, x, data, CONFIG)/CONFIG["num_drop_evals"]
        err = 0
        with torch.no_grad():
            for _ in range(CONFIG["num_drop_evals"]):
                err += (model(x).squeeze()-f).pow(2).mean()/CONFIG["num_drop_evals"]
        lss.backward()
        optimiser.step()
        scheduler.step(lss.item())
        history.append([lss.item(), err.item()])
        ETA_calculation(i, CONFIG["epochs"], start_time, lss.item(), err.item())

    print("")
    save_path = os.path.join(folder_path, 'model.pt')
    torch.save(model.state_dict(), save_path)
    log_path = os.path.join(folder_path, 'loss.csv')
    pd.DataFrame(history, columns=["Loss", "Error"]).to_csv(log_path, index=False)
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w') as blah:
        json.dump(CONFIG, blah, indent=4)
    final_lss = 0
    final_err = 0
    for _ in range(CONFIG["num_drop_evals"]):
        final_lss += loss(model, x, data, CONFIG)/CONFIG["num_drop_evals"]
        final_err += (model(x).squeeze()-f).pow(2).mean()/CONFIG["num_drop_evals"]
    return final_lss.detach(), final_err.detach(), model, x

def test_output():
    final_loss, final_error, model, x = train_model()
    print('final loss', final_loss.item(), 'final error', final_error.item())
    f = 0
    for _ in range(100):
        f += model(x).squeeze()/100
    ef= exact_f(x)
    xc = (x[:,0] * torch.cos(x[:,1])).detach().cpu().numpy()
    yc = (x[:,0] * torch.sin(x[:,1])).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    sc0 = axes[0,0].scatter(xc, yc, c=f.detach().cpu().numpy())
    axes[0,0].set_title("Estimated Solution")
    axes[0,0].axis("equal")
    fig.colorbar(sc0, ax=axes[0,0])

    sc1 = axes[0,1].scatter(xc, yc, c=ef.detach().cpu().numpy())
    axes[0,1].set_title("Exact Solution")
    axes[0,1].axis("equal")
    fig.colorbar(sc1, ax=axes[0,1])

    sc2 = axes[1,0].scatter(xc, yc, c=(f - ef).abs().detach().cpu().numpy())
    axes[1,0].set_title("Absolute Error")
    axes[1,0].axis("equal")
    fig.colorbar(sc2, ax=axes[1,0])

    C = ((f - ef).abs() / (1e-4*ef.abs().max() + ef.abs())).detach().cpu().numpy()
    sc3 = axes[1,1].scatter(xc, yc, c= np.log10(C))
    axes[1,1].set_title("Relative Error")
    axes[1,1].axis("equal")
    fig.colorbar(sc3, ax=axes[1,1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_output()
