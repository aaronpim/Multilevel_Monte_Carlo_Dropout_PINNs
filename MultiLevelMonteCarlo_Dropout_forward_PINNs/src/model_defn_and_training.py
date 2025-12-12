import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import copy
import shutil
import pandas as pd
import sys
from datetime import datetime

# -------------------------------------------------------
# Ensure we can import config.py from the ROOT directory
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # root/src
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # root/
sys.path.append(PROJECT_ROOT)

from config import CONFIG


# Set deterministic behavior
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PINN(nn.Module):
    def __init__(self, CONFIG):
        super(PINN, self).__init__()
        layers = CONFIG["layers"]
        dropout_rate = CONFIG["dropout_rate"]

        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # No activation or dropout on output layer
                layer_list.append(nn.SiLU())
                layer_list.append(nn.Dropout(p=dropout_rate))

        self.model = nn.Sequential(*layer_list)
    def forward(self, x):
        return self.model(x)

def loss_PINNs(model, x, CONFIG):
    residual = 0.0
    for _ in range(CONFIG["num_dropout_repeats"]):
        u = model(x)
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        residual += u - (CONFIG["epsilon"])**2 * d2u - 1
    residual /= CONFIG["num_dropout_repeats"]
    loss_pinns = torch.mean(residual**2)
    return loss_pinns

def loss_BCs(model, x_boundary, u_boundary, CONFIG):
    bc_residual = 0.0
    for _ in range(CONFIG["num_dropout_repeats"]):
        bc_residual += model(x_boundary) - u_boundary
    bc_residual /= CONFIG["num_dropout_repeats"]
    loss_bc = CONFIG["boundary_coeff"]*torch.mean(bc_residual**2)
    return loss_bc, bc_residual

def loss_lag(bc_residual, lagrange_multiplier, CONFIG):
    return torch.sum(bc_residual * lagrange_multiplier)

def define_x_domain(CONFIG, device):
    x = torch.linspace(0, 1, CONFIG["num_x_points"]).view(-1, 1).to(device)
    x.requires_grad = True
    # Boundary conditions
    x_boundary = torch.tensor([[0.0], [1.0]], requires_grad=True).to(device)
    u_boundary = torch.tensor([[0.0], [0.0]]).to(device)
    lagrange_multiplier = torch.tensor([[0.0], [0.0]], device=device)
    return x, x_boundary, u_boundary, lagrange_multiplier

def exact_solution(CONFIG, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    x, _, _, _ = define_x_domain(CONFIG, device)
    x = x.detach()
    exp_eps_inv = torch.exp(torch.tensor([1/CONFIG["epsilon"]])).to(device)
    exact = 1 - (torch.exp(x/CONFIG["epsilon"]) + torch.exp((1 - x)/CONFIG["epsilon"])) / (1 + exp_eps_inv)
    return exact

def train(CONFIG, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), out_dir_name="results"):
    run_dir, config_copy_path = create_directory(out_dir_name)

    model = PINN(CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG["lr_schedule_step"], gamma=CONFIG["lr_gamma"])

    x, x_boundary, u_boundary, lagrange_multiplier = define_x_domain(CONFIG, device)

    best_loss = float("inf")
    best_state = None
    loss_history = []

    for epoch in range(CONFIG["epochs"]):

        optimizer.zero_grad()
        model.train()

        loss_pde = loss_PINNs(model, x, CONFIG)
        loss_bc, bc_residual = loss_BCs(model, x_boundary, u_boundary, CONFIG)
        loss_uz = loss_lag(bc_residual, lagrange_multiplier, CONFIG)

        loss = loss_pde + loss_bc + loss_uz

        # Uzawa update
        if epoch % CONFIG["uzawa_epoch"] == 0:
            with torch.no_grad():
                lagrange_multiplier = lagrange_multiplier + CONFIG["uzawa_coeff"] * bc_residual.detach()

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        total_loss = loss.item()
        loss_history.append([epoch, total_loss, loss_pde.item(), loss_bc.item(), loss_uz.item()])

        # Track best model
        if total_loss < best_loss:
            best_loss = total_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={total_loss:.6f}, PDE={loss_pde.item():.6f}, BC={loss_bc.item():.6f}, Uzawa={loss_uz.item():.6f}")


    model.load_state_dict(best_state)
    model.eval()

    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    loss_csv_path = os.path.join(run_dir, "loss.csv")
    pd.DataFrame(loss_history, columns=["Epoch", "Total", "PDE", "BC", "Uzawa"]).to_csv(loss_csv_path, index=False)

    print("\nTraining complete.")
    print(f"Best model saved to:    {model_path}")
    print(f"Loss history saved to: {loss_csv_path}")
    print(f"Config saved to:        {config_copy_path}")

    return model

def unique_run_name():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"

def create_directory(out_dir_name, root_dir=None):
    if root_dir is None:
        root_dir = PROJECT_ROOT

    out_dir = os.path.join(root_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    run_id = unique_run_name()
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Copy config.py from ROOT directory (not from src/)
    config_src = os.path.join(PROJECT_ROOT, "config.py")
    config_dst = os.path.join(run_dir, "config.py")
    shutil.copy(config_src, config_dst)

    return run_dir, config_dst


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    train(CONFIG)
