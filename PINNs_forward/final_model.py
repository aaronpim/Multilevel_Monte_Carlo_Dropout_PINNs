import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.model_defn import load_PirateNet
from src.trainmodel import train_model, CONFIG_to_folder_path
from src.loss_defn import pinns_loss, bcs_loss, estimate_error, exact_soln, gen_x_and_eps

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "input_dim": 2,
        "hidden_dim": 48,
        "num_blocks": 7,
        "output_dim": 1,
        "p_drop": 0.01,
        "activation": "nn.SiLU()",
        "sigma": 1.0,
        "x_num": 101,
        "eps_num": 301,
        "eps_min": -8,
        "eps_max": 4,
        "BC_coef": 1.0,
        "lr": 1e-3,
        "epochs": 5000,
        }
    CONFIG.update(overwrites)
    return CONFIG

def final_model_train(num_evals = 51, base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu', N_dropout = 100):
    os.makedirs(base_dir, exist_ok=True)
    output_vec = []
    for seed in range(num_evals):
        CONFIG = make_config(seed = seed)
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
        print([seed, final_loss.item(), final_error.item()])
        output_vec.append([seed, final_loss.item(), final_error.item()])
        print("")
        print(f"Completed seed {seed}")
    log_path = base_dir +'/final_models.csv'
    pd.DataFrame(output_vec, columns=["seed", "loss", "error"]).to_csv(log_path, index=False)

def final_model_eval(num_evals = 51, base_dir = 'plots', device ='cuda' if torch.cuda.is_available() else 'cpu', N_dropout = 100):
    os.makedirs(base_dir, exist_ok=True)
    with torch.no_grad():
        for seed in range(num_evals):
            seed_vec  = []
            log_path = base_dir +f'/final_eval_{seed}.pt'
            CONFIG = make_config(seed = seed)
            input_pairs, X, E, _, _ = gen_x_and_eps(CONFIG, device = device)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path) and not os.path.exists(log_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only = True))
                model.train()
                for _ in range(N_dropout):
                    outputs = model(input_pairs).squeeze()
                    exact   = exact_soln(X,E)
                    seed_vec.append(torch.stack([outputs, torch.abs(outputs - exact), torch.abs(outputs - exact)/(1e-12+torch.abs(exact))] ))
                seed_vec = torch.stack(seed_vec)
                torch.save(seed_vec, log_path)
                print("")
                print(f"Completed seed {seed}")

def final_model_plot_mean(num_evals = 51, base_dir = 'plots', device = 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    CONFIG = make_config()
    with torch.no_grad():
        _, X, E, _, _ = gen_x_and_eps(CONFIG, device = device)
        U_mod = 0*X.detach()
        Error = 0*X.detach()
        RelEr = 0*X.detach()
        for seed in range(num_evals):
            log_path = base_dir +f'/final_eval_{seed}.pt'
            seed_vec = torch.load(log_path, weights_only = True)
            U_mod += seed_vec.mean(dim = 0)[0].cpu()/num_evals
            Error += seed_vec.mean(dim = 0)[1].cpu()/num_evals
            RelEr += seed_vec.mean(dim = 0)[2].cpu()/num_evals
        U_mod = U_mod.detach().numpy()
        Error = Error.detach().numpy()
        RelEr = RelEr.detach().numpy()

    plt.figure(figsize=(8, 6))

    plt.pcolor(X.detach(), E.detach(), U_mod)
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$\log(\epsilon)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(base_dir +f'/final_means.pdf')
    plt.close()

    plt.pcolor(X.detach(), E.detach(), Error, norm=colors.LogNorm(vmin=Error.min(), vmax=Error.max()) )
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$\log(\epsilon)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(base_dir +f'/final_error.pdf')
    plt.close()

    plt.pcolor(X[1:-1].detach(), E[1:-1].detach(), RelEr[1:-1], norm=colors.LogNorm(vmin=RelEr[1:-1].min(), vmax=RelEr[1:-1].max()) )
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$\log(\epsilon)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(base_dir +f'/final_rel_error.pdf')
    plt.close()

def final_model_plot_var(num_evals = 51, base_dir = 'plots', device = 'cpu'):
    os.makedirs(base_dir, exist_ok=True)
    CONFIG = make_config()
    with torch.no_grad():
        _, X, E, _, _ = gen_x_and_eps(CONFIG, device = device)
        U_var = 0*X.detach()
        for seed in range(num_evals):
            log_path = base_dir +f'/final_eval_{seed}.pt'
            seed_vec = torch.load(log_path, weights_only = True)
            U_var += seed_vec.var(dim = 0)[0].cpu()/num_evals
        U_var = U_var.detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.pcolor(X.detach(), E.detach(), U_var, norm=colors.LogNorm(vmin=U_var.min(), vmax=U_var.max()) )
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$\log(\epsilon)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(base_dir +f'/final_vars.pdf')
    plt.close()

    with torch.no_grad():
        output_vec  = []
        for seed in range(num_evals):
            CONFIG = make_config(seed = seed)
            input_pairs, X, E, _, _ = gen_x_and_eps(CONFIG, device = device)
            folder_path = CONFIG_to_folder_path(CONFIG, base_dir = 'runs')
            save_path = os.path.join(folder_path, 'model.pt')
            if os.path.exists(save_path):
                model = load_PirateNet(CONFIG, device = device)
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only = True))
                model.eval()
                outputs = model(input_pairs).squeeze()
                output_vec.append(outputs)
        output_vec = torch.stack(output_vec)
        epi_var = output_vec.var(dim = 0).detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.pcolor(X.detach(), E.detach(), epi_var, norm=colors.LogNorm(vmin=epi_var.min(), vmax=epi_var.max()) )
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$\log(\epsilon)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(base_dir +f'/final_epi_var.pdf')
    plt.close()
if __name__ == "__main__":
    # final_model_eval()
    # final_model_plot_mean()
    final_model_plot_var()
