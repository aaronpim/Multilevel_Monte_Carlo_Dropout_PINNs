import os
import time
import torch
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from src.trainmodel import CONFIG_to_folder_path, load_all, train_model
from src.MLMC_eval import get_estimator

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "holdout": 5,
        "normalise": False,
        "log": False,
        "eps": 1e-12,
        "num_hid_layers": 4,
        "hid_dim": 96,
        "activation": "nn.ReLU()",
        "dropout_prob": 0.005,
        "epochs": 10000,
        "lr": 1e-4,
        "power": 2,
        }
    CONFIG.update(overwrites)
    return CONFIG

def varmain(CONFIG, dropout_evals = 100, aleatoric = True, epistemic = False):
    model, values, data, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
    model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
    model.train()
    datavar = data.var(dim = [0,1])
    if aleatoric and not epistemic:
        model.eval()
        outputvar = model(values).var(dim = 0)
    elif not aleatoric and epistemic:
        default_input = torch.zeros(1,2)
        eval_vec = []
        for _ in range(dropout_evals):
            eval_vec.append( model(default_input) )
        eval_vec  = torch.stack(eval_vec).flatten(start_dim=0, end_dim=1)
        outputvar = eval_vec.var(dim = 0)
    elif aleatoric and epistemic:
        dropout_evals = dropout_evals//10
        eval_vec = []
        for _ in range(dropout_evals):
            eval_vec.append( model(values) )
        eval_vec  = torch.stack(eval_vec).flatten(start_dim=0, end_dim=1)
        outputvar = eval_vec.var(dim = 0)
    integration_factor = torch.tensor([(2**3)/(3**3)]).to('cpu').float()
    intvar = integration_factor*torch.sum(outputvar)
    errvar = integration_factor*torch.sum(torch.abs(outputvar-datavar))
    return intvar.item(), errvar.item()

def final_model_train(num_evals = 21):
    output_vec = []
    var_vec = []
    for seed in range(num_evals):
        CONFIG = make_config(seed = seed)
        final_loss, final_hold_loss = train_model(CONFIG)
        output_vec.append([seed, final_loss.item(), final_hold_loss.item()])

        alevar, alevar_err = varmain(CONFIG, aleatoric = True, epistemic = False)
        epivar, _ = varmain(CONFIG, aleatoric = False, epistemic = True)
        fullvar, var_err = varmain(CONFIG, aleatoric = True, epistemic = True)
        var_vec.append([seed, alevar, alevar_err, epivar, fullvar, var_err])

        print("")
        # print(f"Completed seed {seed}")
        print("")
    log_path = 'plots/final_models.csv'
    pd.DataFrame(output_vec, columns=["seed", "train_loss", "hold_loss"]).to_csv(log_path, index=False)
    log_path = 'plots/final_vars.csv'
    pd.DataFrame(var_vec, columns=["seed", "aleatoric variance", "aleatoric variance error", "epistemic variance", "variance", "variance error"]).to_csv(log_path, index=False)

def generic_plot(x,y,z,data, filename, min_alpha = 0.0, alpha_power = 1.0, alpha_reverse = False, clamp = 1e-2):
    xf = x.flatten().numpy()
    yf = y.flatten().numpy()
    zf = z.flatten().numpy()
    try:
        values = data.flatten().numpy()
    except:
        values = data.flatten()

    vmin = max(values.min(), clamp*values.max() )
    vmax = values.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    values_color = norm(values)

    # --- linear normalization for alpha ---
    norm_lin = Normalize(vmin=values.min(), vmax=values.max())
    values_alpha = norm_lin(values)

    # Colormap
    cmap = cm.get_cmap("jet")

    # RGBA colors (log-scaled colors)
    colors = cmap(values_color)

    # Alpha mapping (UNCHANGED behavior)
    if alpha_reverse:
        values_alpha = 1 - values_alpha
    alphas = min_alpha + (1 - min_alpha) * values_alpha**alpha_power

    # Inject alpha into colors
    colors[:, 3] = alphas

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        xf,
        yf,
        zf,
        color=colors,
        s=5
    )

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    # Separate colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_estimators(num_models = 21, base_dir = 'model_plots'):
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.join(base_dir, 'estimator_vec' + '.pt' )
    if not os.path.exists(filename):
        estimator_vec = []
        for seed in range(num_models):
            CONFIG = make_config(seed = 0)
            model, values, _, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
            model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
            model.train()
            exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var = get_estimator(model, values, fidelities = [2,4,8,16,32], levels = [9,7,5,3,1])

            estimator_vec.append(torch.stack([
                exp_estimator_exp.mean(dim = 0),
                exp_estimator_var.mean(dim = 0),
                var_estimator_exp.mean(dim = 0),
                var_estimator_var.mean(dim = 0)
                ]))
        estimator_vec = torch.stack(estimator_vec)
        torch.save(estimator_vec,filename)
    else:
        estimator_vec = torch.load(filename, weights_only = True)
    x = torch.load('Data/X.pt', weights_only = True)
    y = torch.load('Data/Y.pt', weights_only = True)
    z = torch.load('Data/Z.pt', weights_only = True)

    generic_plot(x,y,z, estimator_vec.mean(dim = 0)[0], base_dir + '/exp_estimator_exp.png', alpha_power = 2.0)
    generic_plot(x,y,z, estimator_vec.mean(dim = 0)[1], base_dir + '/exp_estimator_var.png', alpha_power = 0.25)
    generic_plot(x,y,z, estimator_vec.mean(dim = 0)[2], base_dir + '/var_estimator_exp.png', alpha_power = 0.25)
    generic_plot(x,y,z, estimator_vec.mean(dim = 0)[3], base_dir + '/var_estimator_var.png', alpha_power = 0.2)

if __name__ == "__main__":
    final_model_train()
    plot_estimators()
