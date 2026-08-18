import os
import copy
import json
import pandas as pd
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model_defn import load_model


def default_config():
    CONFIG = {
        "seed": 0,
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 128,
        "drop_p": 0.1,
        "num_conv_layers": 3,
        "width" : 32,
        "MLP_width": 128,
        "kernal_size": 3,
        "padding": 1
        }
    return CONFIG

def set_seed(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_all(CONFIG = None, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    if CONFIG is None:
        CONFIG = default_config()
    set_seed(seed = CONFIG["seed"])
    model   = load_model(CONFIG, device = device)
    dataset = torch.load('data/mnist.pt', weights_only = True)
    dataset["train_images"] = dataset["train_images"].unsqueeze(1).float().to(device) / 255.0
    dataset["test_images"]  = dataset["test_images"].unsqueeze(1).float().to(device) / 255.0
    dataset["train_labels"] = dataset["train_labels"].to(device)
    dataset["test_labels"]  = dataset["test_labels"].to(device)
    train_dataset = TensorDataset(dataset["train_images"], dataset["train_labels"])
    train_loader  = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_dataset  = TensorDataset(dataset["test_images"], dataset["test_labels"])
    test_loader   = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    return model, train_loader, test_loader

def CONFIG_to_folder_path(CONFIG = None, base_dir = 'runs'):
    if CONFIG is None:
        CONFIG = default_config()
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

def train_model(CONFIG = None, model = None, dataset = None, device = 'cuda' if torch.cuda.is_available() else 'cpu', base_dir = 'runs'):
    if CONFIG is None:
        CONFIG = default_config()

    folder_path = CONFIG_to_folder_path(CONFIG, base_dir = base_dir)
    set_seed(CONFIG["seed"])

    if model is None or dataset is None:
        model, train_loader, test_loader = load_all(CONFIG = CONFIG, device = device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    history   = []
    start_time= time.time()
    best_loss = float('inf')
    for i in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        optimiser.step()

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
            test_loss = running_loss / len(test_loader.dataset)
            history.append([train_loss, test_loss])
            if test_loss < best_loss:
                best_loss = test_loss
                best_dict = copy.deepcopy(model.state_dict())
            ETA_calculation(i, CONFIG["epochs"], start_time, train_loss, test_loss, best_loss)
    model.load_state_dict(best_dict)
    save_path = os.path.join(folder_path, 'model.pt')
    torch.save(model.state_dict(), save_path)
    log_path = os.path.join(folder_path, 'loss.csv')
    pd.DataFrame(history, columns=["train_loss", "hold_loss"]).to_csv(log_path, index=False)
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    with torch.no_grad():
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
        final_loss = running_loss / len(train_loader.dataset)
        model.eval()
        running_loss = 0.0
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
        final_hold_loss = running_loss / len(test_loader.dataset)
    print("")
    return final_loss, final_hold_loss

if __name__ == "__main__":
    train_model()
