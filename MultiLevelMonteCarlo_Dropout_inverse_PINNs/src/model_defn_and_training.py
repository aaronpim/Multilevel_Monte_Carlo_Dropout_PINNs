import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import copy
import shutil
import pandas as pd
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # final layer has 2 outputs and no activation after it
            if m.out_features == 2:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            else:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(m.bias)

    def __init__(self, CONFIG):
        super(PINN, self).__init__()
        layers = CONFIG["layers"]
        dropout_rate = CONFIG["dropout_rate"]

        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                layer_list.append(nn.Tanh())
                layer_list.append(nn.Dropout(p=dropout_rate))
        self.shared_layers = nn.Sequential(*layer_list)

        hidden = layers[-1]
        self.u_layers = nn.Sequential(
            nn.Linear(hidden, 1)
        )
        self.f_layers = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden, 1)
        )
        self.apply(PINN.init_weights)

    def forward(self, x):
        h = self.shared_layers(x)
        u = self.u_layers(h)
        f = self.f_layers(h)
        cond = x * (1 - x)

        u = u * cond
        f = f * cond
        return u, f

def ell2(y):
    return torch.mean(torch.square(y))

def loss_total(model, x, lagrange_multiplier, CONFIG, target = None):
    if target is None:
        target = (1 + CONFIG["alpha"]*(torch.pi)**4)*torch.sin((torch.pi)*x)

    total_loss = 0.0
    loss_his_accum = np.zeros(6)

    for _ in range(CONFIG["num_dropout_repeats"]):
        u, f = model(x)
        du   = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u  = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        res  = d2u + f

        if CONFIG["global_or_local"] == 'global':
            eps = 1 + torch.rand([])*CONFIG["random_eps_width"] - 0.5*CONFIG["random_eps_width"]
        else:
            eps = 1 + (torch.rand_like(x) * CONFIG["random_eps_width"] - 0.5*CONFIG["random_eps_width"])

        loss_tgt    = 0.5 * ell2(u - target*eps.detach())
        loss_f      = (CONFIG["alpha"]/2) * 0.5 * ell2(f)
        loss_d2u    = (CONFIG["alpha"]/2) * 0.5 * ell2(d2u)
        loss_res    = (CONFIG["beta"]/2) * ell2(res)
        loss_lag    = torch.mean(res*lagrange_multiplier)

        loss = loss_tgt + loss_f + loss_d2u + loss_res + loss_lag
        total_loss += loss
        loss_his_accum += np.array([loss.item(), loss_tgt.item(), loss_f.item(), loss_d2u.item(), loss_res.item(), loss_lag.item()])

    total_loss /= CONFIG["num_dropout_repeats"]
    loss_his_accum /= CONFIG["num_dropout_repeats"]

    return total_loss, loss_his_accum.tolist()

def define_x_domain(CONFIG, device):
    x = torch.linspace(0, 1, CONFIG["num_x_points"]).view(-1, 1).to(device)
    lagrange_multiplier = torch.zeros_like(x, device=device)
    x.requires_grad = True
    return x, lagrange_multiplier

def train(CONFIG, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), out_dir_name="results"):
    run_dir, config_copy_path = create_directory(out_dir_name)

    model = PINN(CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    x, lagrange_multiplier = define_x_domain(CONFIG, device)
    loss_history = []

    start_time = time.time()
    u_exact = (torch.sin(torch.pi*x)).detach()
    f_exact = torch.pi**2 * u_exact

    for epoch in range(CONFIG["epochs"]):

        optimizer.zero_grad()
        model.train()
        loss, loss_his = loss_total(model, x, lagrange_multiplier, CONFIG)

        # Uzawa update
        # Uzawa update
        if epoch % CONFIG["uzawa_epoch"] == 0:
            U = 0
            F = 0
            for _ in range(CONFIG["lagrange_multiplier_reps"]):
                u, f = model(x)
                U += u.detach()/CONFIG["lagrange_multiplier_reps"]
                F += f.detach()/CONFIG["lagrange_multiplier_reps"]
                du   = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                d2u  = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=False)[0]
                res  = d2u + f
                lagrange_multiplier = lagrange_multiplier + CONFIG["uzawa_coeff"] * res.detach()/CONFIG["lagrange_multiplier_reps"]
            error_u = ell2(U-u_exact)/ell2(u_exact)
            error_f = ell2(F-f_exact)/ell2(f_exact)
        loss.backward()
        optimizer.step()
        loss_history.append([epoch] + loss_his + [error_u, error_f])

        # --------------------------
        # ETA calculation using time
        # --------------------------
        if epoch % CONFIG["uzawa_epoch"] == 0:

            elapsed = time.time() - start_time
            epochs_done = epoch + 1
            epochs_total = CONFIG["epochs"]

            avg_sec_per_epoch = elapsed / max(1, epochs_done)
            remaining_sec = avg_sec_per_epoch * (epochs_total - epochs_done)

            # nice HH:MM:SS formatting
            def format_time(t):
                t = int(t)
                h = t // 3600
                m = (t % 3600) // 60
                s = t % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            elapsed_str = format_time(elapsed)
            eta_str = format_time(remaining_sec)
            print(
                f"Epoch {epoch}: Total={loss_his[0]:.4e}, Target={loss_his[1]:.4e}, "
                f"f={loss_his[2]:.4e}, d2u={loss_his[3]:.4e}, "
                f"Residual={loss_his[4]:.4e}, Lag={loss_his[5]:.4e} | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                f"u error: {error_u} | f error: {error_f}"
            )


    model.eval()
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    loss_csv_path = os.path.join(run_dir, "loss.csv")
    pd.DataFrame(loss_history, columns=["Epoch", "Total", "Target", "f loss", "d2u", "Residual", "Lagrangian", "u error", "f error"]).to_csv(loss_csv_path, index=False)

    print("\nTraining complete.")
    print(f"Best model saved to:    {model_path}")
    print(f"Loss history saved to: {loss_csv_path}")
    print(f"Config saved to:        {config_copy_path}")

    return model, run_dir

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

    # Copy this file (model_defn_and_training.py) into the run directory
    current_file_src = os.path.abspath(__file__)
    current_file_dst = os.path.join(run_dir, "model_defn_and_training.py")
    shutil.copy(current_file_src, current_file_dst)

    return run_dir, config_dst

def exact_solution(CONFIG, device  = torch.device("cuda" if torch.cuda.is_available() else "cpu"),  u_or_f = 'f'):
    x, _, = define_x_domain(CONFIG, device)
    x = x.detach()
    output = torch.sin( (torch.pi) * x )
    if u_or_f == 'f':
        output = (torch.pi**2)*output
    return output

if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    model, run_dir = train(CONFIG)
    model.train()
    x, _ = define_x_domain(CONFIG, "cuda")
    # Store predictions
    u_samples = []
    f_samples = []
    d2u_samples = []

    for _ in range(10000):
        u_pred, f_pred = model(x)
        du   = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        d2u  = -1*torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=False)[0]
        u_samples.append(u_pred.detach().cpu().numpy())
        f_samples.append(f_pred.detach().cpu().numpy())
        d2u_samples.append(d2u.detach().cpu().numpy())

    # Convert to arrays of shape (num_samples, num_points)
    u_samples = np.array(u_samples)
    f_samples = np.array(f_samples)
    d2u_samples = np.array(d2u_samples)

    # Compute 25%, 50%, 75% quantiles along the sampling axis (axis=0)
    [f_q25, f_q50, f_q75] = np.quantile(f_samples, [0.25, 0.5, 0.75] , axis=0)
    [u_q25, u_q50, u_q75] = np.quantile(u_samples, [0.25, 0.5, 0.75] , axis=0)
    [d_q25, d_q50, d_q75] = np.quantile(d2u_samples, [0.25, 0.5, 0.75] , axis=0)

    # Exact solutions
    u_exact = (torch.sin(torch.pi*x)).detach().cpu().numpy()
    f_exact = (torch.pi**2 * torch.sin(torch.pi*x)).detach().cpu().numpy()

    # Plot f(x)
    plt.fill_between(x.detach().cpu().numpy().flatten(), f_q25.flatten()-f_exact.flatten(), f_q75.flatten()-f_exact.flatten(), color='lightblue', alpha=0.5, label='25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), f_q50.flatten()-f_exact.flatten(), color='blue', label='50% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), f_exact.flatten()-f_exact.flatten(), 'k--', label='Exact')
    plt.plot(x.detach().cpu().numpy().flatten(), (1+CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:',  label='Exact 25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), (1-CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:')
    plt.legend()
    plt.title('f plot')
    plt.savefig(os.path.join(run_dir, "f_plot.pdf"), bbox_inches='tight')
    plt.close()

    plt.fill_between(x.detach().cpu().numpy().flatten(), d_q25.flatten()-f_exact.flatten(), d_q75.flatten()-f_exact.flatten(), color='lightblue', alpha=0.5, label='25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), d_q50.flatten()-f_exact.flatten(), color='blue', label='50% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), f_exact.flatten()-f_exact.flatten(), 'k--', label='Exact')
    plt.plot(x.detach().cpu().numpy().flatten(), (1+CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:',  label='Exact 25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), (1-CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:')
    plt.legend()
    plt.title('d2u plot')
    plt.savefig(os.path.join(run_dir, "d2u_plot.pdf"), bbox_inches='tight')
    plt.close()

    plt.fill_between(x.detach().cpu().numpy().flatten(), u_q25.flatten()-u_exact.flatten(), u_q75.flatten()-u_exact.flatten(), color='lightblue', alpha=0.5, label='25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), u_q50.flatten()-u_exact.flatten(), color='blue', label='50% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), u_exact.flatten()-u_exact.flatten(), 'k--', label='Exact')
    plt.plot(x.detach().cpu().numpy().flatten(), (1+CONFIG["random_eps_width"]/4)*u_exact.flatten()-u_exact.flatten(), 'k:',  label='Exact 25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), (1-CONFIG["random_eps_width"]/4)*u_exact.flatten()-u_exact.flatten(), 'k:')
    plt.legend()
    plt.title('u plot')
    plt.savefig(os.path.join(run_dir, "u_plot.pdf"), bbox_inches='tight')
    plt.close()

    #############################################
    plt.fill_between(x.detach().cpu().numpy().flatten(), f_q25.flatten()-f_q50.flatten(), f_q75.flatten()-f_q50.flatten(), color='lightgreen', alpha=0.5, label='25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), f_exact.flatten()-f_exact.flatten(), 'k--', label='Exact')
    plt.plot(x.detach().cpu().numpy().flatten(), (1+CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:',  label='Exact 25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), (1-CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:')
    plt.legend()
    plt.title('f plot')
    plt.savefig(os.path.join(run_dir, "f_IQR_plot.pdf"), bbox_inches='tight')
    plt.close()

    plt.fill_between(x.detach().cpu().numpy().flatten(), d_q25.flatten()-d_q50.flatten(), d_q75.flatten()-d_q50.flatten(), color='lightgreen', alpha=0.5, label='25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), f_exact.flatten()-f_exact.flatten(), 'k--', label='Exact')
    plt.plot(x.detach().cpu().numpy().flatten(), (1+CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:',  label='Exact 25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), (1-CONFIG["random_eps_width"]/4)*f_exact.flatten()-f_exact.flatten(), 'k:')
    plt.legend()
    plt.title('d2u plot')
    plt.savefig(os.path.join(run_dir, "d2u_IQR_plot.pdf"), bbox_inches='tight')
    plt.close()

    plt.fill_between(x.detach().cpu().numpy().flatten(), u_q25.flatten()-u_q50.flatten(), u_q75.flatten()-u_q50.flatten(), color='lightgreen', alpha=0.5, label='25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), u_exact.flatten()-u_exact.flatten(), 'k--', label='Exact')
    plt.plot(x.detach().cpu().numpy().flatten(), (1+CONFIG["random_eps_width"]/4)*u_exact.flatten()-u_exact.flatten(), 'k:',  label='Exact 25-75% quantile')
    plt.plot(x.detach().cpu().numpy().flatten(), (1-CONFIG["random_eps_width"]/4)*u_exact.flatten()-u_exact.flatten(), 'k:')
    plt.legend()
    plt.title('u plot')
    plt.savefig(os.path.join(run_dir, "u_IQR_plot.pdf"), bbox_inches='tight')
    plt.close()
    if False:
        # train(CONFIG)
        model = PINN(CONFIG)
        x = torch.linspace(0, 1, CONFIG["num_x_points"], requires_grad = True).view(-1, 1)
        print('x shape:', x.shape)
        u, f = model(x)
        print('u shape:', u.shape)
        print('f shape:',f.shape)
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        print('du shape:',du.shape)
        print('d2u shape:',d2u.shape)
        print('u endpoints to check BCs:',u[0], u[-1])
        ###
        x, lagrange_multiplier = define_x_domain(CONFIG, "cpu")
        loss_test, res = loss(model, x, lagrange_multiplier, CONFIG)
        print(loss_test)
        loss_test.backward()
