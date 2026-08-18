import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.MLMC_base import make_config, multi_level_multi_fidelity_eval, levels_to_num_evals
from src.trainmodel import load_all, compute_delaunay_weights, set_seed, CONFIG_to_folder_path

def compute_M(T, c):
    s = sum((T[i] - T[i - 1]) / np.sqrt(T[i] * T[i - 1]) for i in range(1, len(T)))
    factor = 1.0 / (1.0 + s)
    M = [int(np.ceil(c / T[0] * factor))]
    for i in range(1, len(T)):
        M.append(int(np.ceil(c / np.sqrt(T[i] * T[i - 1]) * factor)))
    num_evals = levels_to_num_evals(num_levels = M)
    cost = 0
    for i in range(len(num_evals)):
        cost += T[i]*num_evals[i]
    return M, cost.item()

def get_estimator_variance_vary_cost(num_models = 21, fidelity_ladder =[4, 8, 16], base_cost = 50, cost_multiplier = [1,2,4,8,16], log_path = 'plots/estimator_variance_vary_cost.csv'):
    output_vec  = []
    with torch.no_grad():
        for C in cost_multiplier:
            num_levels, cost = compute_M(T = fidelity_ladder, c = base_cost*C)
            for seed in range(num_models):
                set_seed(seed = seed)
                CONFIG = make_config(seed = seed)
                model, values, _, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
                model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
                model.train()
                values = values[:20]
                weights = compute_delaunay_weights(values, device = 'cpu')
                _, mean_estimator_variance, _, sample_variance_estimator_variance = multi_level_multi_fidelity_eval(model, values, fidelity_ladder = fidelity_ladder, num_levels = num_levels, factor = (8/27), weights = weights)
                output_vec.append([cost, seed, mean_estimator_variance.item(), sample_variance_estimator_variance.item()])
                print(f"Completed seed {seed}, with cost {cost}")
        pd.DataFrame(output_vec, columns=["cost", "seed", "mean estimator variance", "sample variance estimator variance"]).to_csv(log_path, index=False)

def plot_estimator_variance_vary_cost( csv_path="plots/estimator_variance_vary_cost.csv", figsize=(6, 4)):
    df = pd.read_csv(csv_path)
    df_med = (df.groupby("cost", as_index=False).agg({"mean estimator variance": "median", "sample variance estimator variance": "median"}).sort_values("cost"))
    stats = (df.groupby("cost").agg(
            mean_mean=("mean estimator variance", "median"),
            q25_mean=("mean estimator variance", lambda x: x.quantile(0.25)),
            q75_mean=("mean estimator variance", lambda x: x.quantile(0.75)),
            min_mean=("mean estimator variance", "min"),
            max_mean=("mean estimator variance", "max"),
            mean_var=("sample variance estimator variance", "median"),
            q25_var=("sample variance estimator variance", lambda x: x.quantile(0.25)),
            q75_var=("sample variance estimator variance", lambda x: x.quantile(0.75)),
            min_var=("sample variance estimator variance", "min"),
            max_var=("sample variance estimator variance", "max")
        ).reset_index().sort_values("cost"))
    plt.figure(figsize=figsize)
    plt.loglog(df_med["cost"], df_med["mean estimator variance"], marker="o", linewidth=2, color = 'orange')
    plt.fill_between(stats["cost"], stats["min_mean"], stats["max_mean"], color="orange", alpha=0.15)
    plt.fill_between(stats["cost"], stats["q25_mean"], stats["q75_mean"], color="orange", alpha=0.35)
    plt.xlabel("Cost")
    plt.ylabel("Mean estimator variance")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/estimator_variance_vary_cost_mean.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.loglog( df_med["cost"], df_med["sample variance estimator variance"],marker="o", linewidth=2, color = 'orange')
    plt.fill_between(stats["cost"], stats["min_var"], stats["max_var"], color="orange", alpha=0.15)
    plt.fill_between(stats["cost"], stats["q25_var"], stats["q75_var"], color="orange", alpha=0.35)
    plt.xlabel("Cost")
    plt.ylabel("Sample variance estimator variance")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/estimator_variance_vary_cost_var.pdf')
    plt.close()

def estimator_variance_vary_cost_rate( csv_path="plots/estimator_variance_vary_cost.csv"):
    df = pd.read_csv(csv_path)
    df_med = (df.groupby("cost", as_index=False).agg({"mean estimator variance": 'median', "sample variance estimator variance": 'median'}).sort_values("cost"))
    x = np.log(df_med["cost"].to_numpy())
    y_mean = np.log(df_med["mean estimator variance"].to_numpy())
    slope_mean, intercept_mean = np.polyfit(x, y_mean, 1)
    y_var = np.log(df_med["sample variance estimator variance"].to_numpy())
    slope_var, intercept_var = np.polyfit(x, y_var, 1)
    print(f"Mean estimator variance ~ cost^{slope_mean:.4f}")
    print(f"Sample variance estimator variance ~ cost^{slope_var:.4f}")

if __name__ == "__main__":
    get_estimator_variance_vary_cost()
    plot_estimator_variance_vary_cost()
    estimator_variance_vary_cost_rate()
