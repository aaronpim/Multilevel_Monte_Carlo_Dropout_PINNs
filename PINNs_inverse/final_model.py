import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.model_defn import load_model
from src.trainmodel import train_model, CONFIG_to_folder_path, sunflower_disk_points, exact_f

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "epochs": 3000,
        "num_modes": 2,
        "num_lay": 2,
        "hid_dim": 128,
        "smoothing_coef": 0.0,
        "drop_p": 0.001,
        "num_x_points": 64,
        "num_y_points": 10000,
        "num_drop_evals": 20,
        "clamp": 1e-12,
        "lr": 1e-3
        }
    CONFIG.update(overwrites)
    return CONFIG

def final_model_train(num_evals = 101):
    output_vec = []
    for seed in range(num_evals):
        CONFIG = make_config(seed = seed)
        final_lss, final_err, model, x = train_model(CONFIG)
        output_vec.append([seed, final_lss.item(), final_err.item()])
        print("")
        print(f"Completed seed {seed}")
        print("")
    log_path = 'plots/final_models.csv'
    pd.DataFrame(output_vec, columns=["seed", "final loss", "final error"]).to_csv(log_path, index=False)

def final_model_eval(num_eval_points = 4096, num_dropout = 1001, num_models = 101):
    x = sunflower_disk_points(num_eval_points, device = 'cpu')
    dropout_mean_tensor = []
    dropout_var_tensor  = []
    dropout_err_tensor  = []
    with torch.no_grad():
        for seed in range(num_models):
            CONFIG = make_config(seed = seed)
            model  = load_model(CONFIG, device = 'cpu')
            model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
            model.train()
            evals = torch.stack([ model(x) for _ in range(num_dropout) ]).squeeze()
            errs  = (evals - exact_f(x).unsqueeze(0)).abs()
            dropout_mean_tensor.append(evals.mean(dim = 0))
            dropout_var_tensor.append(evals.var(dim = 0))
            dropout_err_tensor.append(errs.mean(dim = 0))
            print(f'Completed seed {seed}/{num_models-1}')
    dropout_mean_tensor = torch.stack(dropout_mean_tensor)
    dropout_var_tensor  = torch.stack(dropout_var_tensor)
    dropout_err_tensor  = torch.stack(dropout_err_tensor)
    torch.save(dropout_mean_tensor, 'plots/dropout_mean_data.pt')
    torch.save(dropout_var_tensor,  'plots/dropout_var_data.pt')
    torch.save(dropout_err_tensor,  'plots/dropout_err_data.pt')

def final_model_plot():
    dropout_mean_tensor = torch.load('plots/dropout_mean_data.pt', weights_only = True)
    dropout_var_tensor  = torch.load('plots/dropout_var_data.pt', weights_only = True)
    dropout_err_tensor  = torch.load('plots/dropout_err_data.pt', weights_only = True)
    inputs = sunflower_disk_points(dropout_mean_tensor.shape[1], device = 'cpu')
    plot_values(inputs, dropout_mean_tensor.mean(dim = 0), 'plots/mean.png', scatter = True)
    plot_values(inputs, dropout_var_tensor.mean(dim = 0), 'plots/var.png')
    plot_values(inputs, dropout_err_tensor.mean(dim = 0), 'plots/error.png')
    plot_values(inputs, (dropout_err_tensor/(1e-4*exact_f(inputs).abs().max()+exact_f(inputs).abs())).mean(dim = 0), 'plots/rel_error.png', log = True)

def epistemic_model_plot(num_eval_points = 4096, num_models = 101):
    inputs = sunflower_disk_points(num_eval_points, device = 'cpu')
    eval_tensor = []
    with torch.no_grad():
        for seed in range(num_models):
            CONFIG = make_config(seed = seed)
            model  = load_model(CONFIG, device = 'cpu')
            model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
            model.eval()
            eval_tensor.append(model(inputs))
    eval_tensor = torch.stack(eval_tensor).squeeze()
    plot_values(inputs, eval_tensor.var(dim = 0), 'plots/epi_var.png', log = True, scatter = True)

def plot_values(inputs, output, name, scatter = False, log = False):
    r = inputs[:, 0].numpy()
    theta = inputs[:, 1].numpy()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if log:
        plt.tripcolor(x, y, output.numpy(), shading="gouraud", cmap="viridis", norm=LogNorm())
    else:
        plt.tripcolor(x, y, output.numpy(), shading="gouraud", cmap="viridis")
    plt.colorbar()
    if scatter:
        inputs = sunflower_disk_points(64, device = 'cpu')
        r = inputs[:, 0].numpy()
        theta = inputs[:, 1].numpy()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.scatter(x,y, color = 'red', marker = 'x', linewidths = 0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    #final_model_train()
    #final_model_eval()
    #final_model_plot()
    epistemic_model_plot()
