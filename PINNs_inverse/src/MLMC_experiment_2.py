import os
import math
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.MLMC_base import make_config, multi_level_multi_fidelity_eval, levels_to_num_evals
from src.trainmodel import set_seed, CONFIG_to_folder_path, sunflower_disk_points
from src.model_defn import load_model
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def reference_mean_and_variance(seed = 0, x_data_points = 4096, batch_size = 1000, num_batches = 500, filename = None):
    '''
    This function uses Welford's algorithm to compute the mean and variance of the model
    and then save the outputs as reference data.
    '''
    if filename is None:
        filename = f'plots/reference_estimator_Nx_{x_data_points}_T_{batch_size*num_batches}.pt'
    with torch.no_grad ():
        CONFIG = make_config(seed = 0)
        model  = load_model(CONFIG, device = 'cpu')
        model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
        model.train()
        values = sunflower_disk_points(x_data_points)
        running_mean = None
        running_M2 = None
        count = 0
        start_time = time.time()
        for i in range(num_batches):
            evals = torch.stack([ model(values) for _ in range(batch_size)])
            batch_mean = evals.mean(dim=0)
            batch_M2 = evals.var(dim=0, unbiased=False) * batch_size
            if running_mean is None:
                running_mean = batch_mean
                running_M2 = batch_M2
                count = batch_size
                continue
            new_count = count + batch_size
            delta = batch_mean - running_mean
            running_mean += delta * batch_size / new_count
            running_M2 += batch_M2 + delta.square() * count * batch_size / new_count
            count = new_count
            print(f'Compeleted batch {i+1}/{num_batches}, ETA: {((num_batches+1)/(i+1) - 1) * (time.time() - start_time)}')
    running_var = running_M2 / count
    torch.save({"mean": running_mean, "variance": running_var}, filename)

def load_reference_mean_and_variance(seed = 0, x_data_points = 4096, batch_size = 1000, num_batches = 500, filename = None):
    if filename is None:
        filename = f'plots/reference_estimator_Nx_{x_data_points}_T_{batch_size*num_batches}.pt'
    data = torch.load(filename, weights_only = True)
    ref_mean = data['mean']
    ref_var  = data['variance']
    return ref_mean, ref_var

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

def fixed_cost_RMSE(master_seed = 0, num_reps = 50, fidelity_ladder =[4, 8, 16], base_cost = 50, cost_multiplier = [1,2,4,8,16], log_paths = ['plots/time_v_cost.csv', 'plots/RMSE_v_cost.csv']):
    with torch.no_grad():
        clock_vec = []
        RMSE_vec  = []
        CONFIG = make_config(seed = master_seed)
        model  = load_model(CONFIG, device = 'cpu')
        model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
        model.train()
        values = sunflower_disk_points(4096)
        ref_mean, ref_sample_var = load_reference_mean_and_variance(seed = master_seed)
        ref_mean = math.pi * ref_mean.mean()
        ref_sample_var = math.pi * ref_sample_var.mean()
        for C in cost_multiplier:
            mean_estimates = []
            var_estimates = []
            num_levels, cost = compute_M(fidelity_ladder, C * base_cost)
            for seed in range(num_reps):
                set_seed(seed = seed)
                start_time = time.time()
                mean_estimator, _, sample_variance_estimator, _ = multi_level_multi_fidelity_eval(model, values, fidelity_ladder = fidelity_ladder, num_levels = num_levels)
                wall_clock_time = time.time() - start_time
                mean_estimates.append(mean_estimator)
                var_estimates.append(sample_variance_estimator)
                clock_vec.append([cost, seed, wall_clock_time])
                print(f"Completed seed {seed}, with cost {cost}")
            mean_estimates = torch.stack(mean_estimates)
            var_estimates  = torch.stack(var_estimates)
            RMSE_mean = (mean_estimates - ref_mean).square().mean().sqrt().item()
            RMSE_var  = (var_estimates - ref_sample_var).square().mean().sqrt().item()
            RMSE_vec.append([cost, RMSE_mean,  RMSE_var])
        pd.DataFrame(clock_vec, columns=["cost", "seed", "time"]).to_csv(log_paths[0], index=False)
        pd.DataFrame(RMSE_vec, columns=["cost", "RMSE mean", "RMSE var"]).to_csv(log_paths[1], index=False)

def plot_RMSE_v_cost(log_path = 'plots/RMSE_v_cost.csv', figsize=(6, 4)):
    df = pd.read_csv(log_path)
    df = df.sort_values("cost")
    plt.figure(figsize=figsize)
    plt.loglog(df["cost"], df["RMSE mean"], marker="o", linewidth=2, color = 'orange')
    plt.xlabel("Cost")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/RMSE_v_cost_mean.pdf')
    plt.close()

    plt.figure(figsize=figsize)
    plt.loglog(df["cost"], df["RMSE var"], marker="o", linewidth=2, color = 'orange')
    plt.xlabel("Cost")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/RMSE_v_cost_var.pdf')
    plt.close()

def plot_time_v_cost(log_path = 'plots/time_v_cost.csv', figsize=(6, 4)):
    df = pd.read_csv(log_path)
    stats = (df.groupby("cost").agg(
            median = ("time", 'median'),
            q66_low=("time", lambda x: x.quantile(0.166666)),
            q66_upp=("time", lambda x: x.quantile(0.833333)),
            q95_low=("time", lambda x: x.quantile(0.025)),
            q95_upp=("time", lambda x: x.quantile(0.975))
        ).reset_index().sort_values("cost"))
    plt.figure(figsize=figsize)
    plt.plot(stats["cost"], stats["median"], marker="o", linewidth=2, color = 'orange')
    plt.fill_between(stats["cost"], stats["q95_low"], stats["q95_upp"], color="orange", alpha=0.15)
    plt.fill_between(stats["cost"], stats["q66_low"], stats["q66_upp"], color="orange", alpha=0.35)
    plt.xlabel("Cost")
    plt.ylabel("Time")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/Time_v_cost.pdf')
    plt.close()

def estimator_variance_vary_cost_rate( csv_path = 'plots/RMSE_v_cost.csv'):
    df = pd.read_csv(csv_path)
    x = np.log(df["cost"].to_numpy())
    y_mean = np.log(df["RMSE mean"].to_numpy())
    slope_mean, intercept_mean = np.polyfit(x, y_mean, 1)
    y_var = np.log(df["RMSE var"].to_numpy())
    slope_var, intercept_var = np.polyfit(x, y_var, 1)
    print(f"Mean estimator variance ~ cost^{slope_mean:.4f}")
    print(f"Sample variance estimator variance ~ cost^{slope_var:.4f}")

def hypothesis_test_linearity(csv_path = 'plots/time_v_cost.csv'):
    df = pd.read_csv(csv_path)
    stats = (df.groupby("cost").agg(median = ("time", 'median')).reset_index().sort_values("cost"))
    m = smf.ols("median ~ cost", data=stats).fit()
    pred = m.predict(stats)
    print(m.rsquared)
    m = smf.ols("time ~ cost", data=df).fit()
    pred = m.predict(df)
    print(m.rsquared)

# Runtime exhibited approximately linear scaling with cost over the tested range. A linear regression explained 99.7% of the variance across all runs (R2=0.9970), while regression on the median runtime at each cost yielded R2=0.9985.

if __name__ == "__main__":
    #reference_mean_and_variance()
    # fixed_cost_RMSE()
    # plot_RMSE_v_cost()
    # plot_time_v_cost()
    # estimator_variance_vary_cost_rate()
    hypothesis_test_linearity()
