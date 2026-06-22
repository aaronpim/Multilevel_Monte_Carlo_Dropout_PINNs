import os
import time
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from src.MLMC_eval import get_estimator
from src.trainmodel import CONFIG_to_folder_path, load_all
from final_model import make_config

def all_levels(cost=256, T=[4,8,16]):
    # Compute weights w_l
    w = [T[0]] + [T[i] - T[i-1] for i in range(1, len(T))]
    L = len(w)
    # Upper bounds for each M_l
    Mmax = [cost // wi for wi in w]
    valid_solutions = []
    # Helper: check non-increasing order
    def is_non_increasing(lst):
        return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))
    def is_grtr_thn_1(lst):
        return all(lst[i] >= 2 for i in range(len(lst)))
    # Enumerate all possibilities
    ranges = [range(m+1) for m in Mmax]
    for M in itertools.product(*ranges):
        total = sum(w[i] * M[i] for i in range(L))
        if total == cost and is_non_increasing(M) and is_grtr_thn_1(M):
            valid_solutions.append(M)
    return valid_solutions

def compute_theoretical_M_exp(T=[4, 8, 16], c_T=256):
    T = np.array(T)
    L = len(T) - 1

    diffs = T[1:] - T[:-1]
    denom_terms = diffs / np.sqrt(T[1:] * T[:-1])
    denominator = 1 + np.sum(denom_terms)


    M0 = c_T / T[0] / denominator
    M_rest = c_T / np.sqrt(T[1:] * T[:-1]) / denominator
    M = np.concatenate(([M0], M_rest))
    return M

def compute_theoretical_M_var(T = [4, 8, 16], c_T = 256):
    T = np.array(T)
    L = len(T) - 1
    base_term = np.sqrt(T[0] / (T[0] - 1))

    diffs = T[1:] - T[:-1]
    denom_terms = diffs / np.sqrt((T[1:] - 1) * (T[:-1] - 1))

    denominator = base_term + np.sum(denom_terms)

    M0 = c_T / np.sqrt(T[0] * (T[0] - 1)) / denominator
    M_rest = c_T / np.sqrt((T[1:] - 1) * (T[:-1] - 1)) / denominator

    return np.concatenate(([M0], M_rest))

def generate_data_fixed_cost(c = 256, T = [4,8,16], base_dir = 'experiments', dir_name = None, num_models = 21, factor = (2/3)**3):
    if dir_name is None:
        dir_name = f'cost_{c}_fids_{T}'
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    M_all = all_levels(cost = c, T=T)
    for seed in range(num_models):
        data_name = f'data_{seed}.pt'
        data_path = os.path.join(dir_path, data_name)
        if os.path.exists(data_path):
            print(f"Results already exist")
        else:
            CONFIG = make_config(seed = seed)
            model, values, _, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
            values = torch.zeros(1,values.shape[1])
            model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
            model.train()
            exp_var = []
            var_var = []
            start_time = time.time()
            for i, M in enumerate(M_all):
                _, exp_estimator_var, _, var_estimator_var = get_estimator(model, values, fidelities = T, levels = M)
                exp_var.append( exp_estimator_var.sum()*factor )
                var_var.append( var_estimator_var.sum()*factor )
                elapsed = time.time() - start_time
                iters_done = i + 1
                total_iters = len(M_all)
                avg_per_iter = elapsed / iters_done
                remaining = avg_per_iter * (total_iters - iters_done)

                print(
                    f"\rCompleted {iters_done} out of {total_iters} "
                    f"— elapsed: {elapsed:.1f}s, "
                    f"ETA: {remaining:.1f}s "
                    f"Seed: {seed}",
                    end="",
                    flush=True
                )
            print()
            data = {
                'M_all': M_all,
                'exp_var': torch.stack(exp_var),
                'var_var': torch.stack(var_var),
                }
            torch.save(data, data_path)

def plot_data_fixed_cost(c = 256, T = [4,8,16], base_dir = 'experiments', dir_name = None, num_models = 21, alpha = 0.005):
    if dir_name is None:
        dir_name = f'cost_{c}_fids_{T}'
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    exp_var = 0
    var_var = 0
    for seed in range(num_models):
        data_name = f'data_{seed}.pt'
        data_path = os.path.join(dir_path, data_name)
        data_i = torch.load(data_path, weights_only = True)
        # exp_var += data_i['exp_var']/num_models
        # var_var += data_i['var_var']/num_models
        exp_var = data_i['exp_var']
        var_var = data_i['var_var']
        exp_var = exp_var.numpy()
        var_var = var_var.numpy()
        M_all = data_i['M_all']
        x = np.array([point[1] for point in M_all])
        y = np.array([point[2] for point in M_all])
        N = max(1, int(alpha * len(exp_var)))
        print("Bottom", N, "samples")
        A = np.sort(exp_var)
        I = np.argsort(exp_var)
        avg_min_M1 = np.mean(x[I[:N]])
        avg_min_M2 = np.mean(y[I[:N]])
        avg_min_z = np.mean(A[:N])
        est_M0_val = (c - avg_min_M2*(T[2]-T[1]) - avg_min_M1*(T[1]-T[0]))/T[0]
        print(f"Emperical Expectation levels, M0 = {est_M0_val:.4f}, M1 = {avg_min_M1:.4f}, M2 = {avg_min_M2:.4f}, Z = {avg_min_z}")
        theo = compute_theoretical_M_exp(T =T, c_T =c)
        print(f"Theoretical Expectation levels, M0 = {theo[0]:.4f}, M1 = {theo[1]:.4f}, M2 = {theo[2]:.4f}")
        plt.figure(figsize=(8,6))
        contour = plt.tricontourf(x, y, 1/exp_var, levels=25, cmap='viridis')
        plt.xlabel(r'$M_1$')
        plt.ylabel(r'$M_2$')
        plt.colorbar(contour, label=r'$\left(\int S_Y^2(x)~ dx\right)^{-1}$')
        plt.tight_layout()
        save_path = os.path.join(dir_path, f"var_exp_estimator_fixed_cost_{seed}.pdf")
        plt.scatter(avg_min_M1, avg_min_M2, c='k', marker = 'x')
        plt.scatter(theo[1], theo[2], c='r', marker = 'x')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


        A = np.sort(var_var)
        I = np.argsort(var_var)
        avg_min_M1 = np.mean(x[I[:N]])
        avg_min_M2 = np.mean(y[I[:N]])
        avg_min_z = np.mean(A[:N])
        est_M0_val = (c - avg_min_M2*(T[2]-T[1]) - avg_min_M1*(T[1]-T[0]))/T[0]
        print(f"Emperical Variance levels, M0 = {est_M0_val:.4f}, M1 = {avg_min_M1:.4f}, M2 = {avg_min_M2:.4f}, Z = {avg_min_z}")
        theo = compute_theoretical_M_var(T =T, c_T =c)
        print(f"Theoretical Variance levels, M0 = {theo[0]:.4f}, M1 = {theo[1]:.4f}, M2 = {theo[2]:.4f}")
        plt.figure(figsize=(8,6))
        contour = plt.tricontourf(x, y, 1/var_var, levels=20, cmap='viridis')
        plt.xlabel(r'$M_1$')
        plt.ylabel(r'$M_2$')
        plt.colorbar(contour, label=r'$\left(\int S_V^2(x)~ dx\right)^{-1}$')
        plt.tight_layout()
        save_path = os.path.join(dir_path, f"var_var_estimator_fixed_cost_{seed}.pdf")
        plt.scatter(avg_min_M1, avg_min_M2, c='k', marker = 'x')
        plt.scatter(theo[1], theo[2], c='r', marker = 'x')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # generate_data_fixed_cost()
    plot_data_fixed_cost()
