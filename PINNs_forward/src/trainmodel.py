import os
import copy
import json
import time
import torch
import pandas as pd
from src.model_defn import load_PirateNet
from src.loss_defn import pinns_loss, bcs_loss, estimate_error

def default_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "input_dim": 2,
        "hidden_dim": 64,
        "num_blocks": 8,
        "output_dim": 1,
        "p_drop": 0.05,
        "activation": "nn.SiLU()",
        "sigma": 1.0,
        "x_num": 101,
        "eps_num": 101,
        "eps_min": -7,
        "eps_max": 3,
        "BC_coef": 1.0,
        "lr": 1e-3,
        "epochs": 5000,
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

def ETA_calculation(epoch, maxepochs, start_time, ploss, bloss, loss, error, best_loss):
    elapsed = time.time() - start_time
    avg_time_per_epoch = elapsed/(epoch+1)
    eta_seconds = avg_time_per_epoch * (maxepochs - epoch)
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    print(f"\rEpoch {epoch+1}/{maxepochs} | ETA: {eta_str} | PINNs Loss: {ploss:.5e} | BC loss: {bloss:.5e} | Loss: {loss:.5e} | Error: {error:.5e} | Best loss: {best_loss:.5e}", end="", flush=True)

def train_model(CONFIG = None, device = 'cuda' if torch.cuda.is_available() else 'cpu', base_dir = 'runs'):
    if CONFIG is None:
        CONFIG = default_config()
    folder_path = CONFIG_to_folder_path(CONFIG, base_dir = base_dir)
    set_seed(CONFIG["seed"])
    model = load_PirateNet(CONFIG, device = device)
    optimiser  = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=200, threshold=1e-5, threshold_mode="rel", cooldown=50, min_lr=1e-7)
    history    = []
    start_time = time.time()
    best_loss  = float('inf')
    for i in range(CONFIG["epochs"]):
        model.train()
        optimiser.zero_grad()
        ploss = pinns_loss(model, CONFIG, device=device)
        bloss = bcs_loss(model, CONFIG, device=device)
        loss  = ploss + CONFIG["BC_coef"] * bloss
        error = estimate_error(model, CONFIG, device = device)
        history.append([ploss.item(), bloss.item(), loss.item(), error.item()])
        if loss.item() < best_loss:
            best_model = copy.deepcopy(model.state_dict())
            best_loss  = loss.item()
        ETA_calculation(i, CONFIG["epochs"], start_time, ploss.item(), bloss.item(), loss.item(), error.item(), best_loss)
        loss.backward()
        optimiser.step()
        scheduler.step(loss.item())
    print("")
    model.load_state_dict(best_model)
    save_path = os.path.join(folder_path, 'model.pt')
    torch.save(model.state_dict(), save_path)
    log_path = os.path.join(folder_path, 'loss.csv')
    pd.DataFrame(history, columns=["PINNs loss", "BC loss", "Loss", "Error"]).to_csv(log_path, index=False)
    config_path = os.path.join(folder_path, 'config.json')

    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    final_loss = 0
    final_error= 0
    N_dropout = 100
    for _ in range(N_dropout):
        final_loss  += pinns_loss(model, CONFIG, device=device).detach()/N_dropout + CONFIG["BC_coef"] * bcs_loss(model, CONFIG, device=device).detach()/N_dropout
        final_error += estimate_error(model, CONFIG, device=device).detach()/N_dropout
    print(['final loss: ', final_loss.item(), 'final error: ',final_error.item()])
    return final_loss, final_error

if __name__ == "__main__":
    final_loss, final_error = train_model()
    print('final loss', final_loss.item(), ' final error', final_error.item() )
