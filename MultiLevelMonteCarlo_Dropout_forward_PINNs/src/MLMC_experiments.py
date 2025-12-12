import os
import json
import time
import hashlib
import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
from scipy.optimize import curve_fit
from MLMC_eval import get_estimator
from Load_Model import load_latest_model
from model_defn_and_training import create_directory, exact_solution

def prepare_experiment_directory(run_dir, fidelities, levels, experiment_number = 1):
    experiment_dir = os.path.join(run_dir, f"experiment_{experiment_number}")
    os.makedirs(experiment_dir, exist_ok=True)

    hash_input = json.dumps({"fidelities": fidelities, "levels": levels}, sort_keys=True)
    unique_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    experiment_subdir = os.path.join(experiment_dir, unique_hash)
    os.makedirs(experiment_subdir, exist_ok=True)

    param_file = os.path.join(experiment_subdir, "parameters.json")
    with open(param_file, "w") as f:
        json.dump({"fidelities": fidelities, "levels": levels}, f, indent=4)

    return experiment_subdir


def initialise(fidelities = [2,4,8,16,32], levels = [9,7,5,3,1]):
    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    exact = exact_solution(CONFIG, device = "cpu")
    exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var = get_estimator(model, x, fidelities, levels)
    x = x.numpy().flatten()
    exact = exact.numpy().flatten()
    exp_estimator_exp = exp_estimator_exp.flatten()
    exp_estimator_var = exp_estimator_var.flatten()
    var_estimator_exp = var_estimator_exp.flatten()
    var_estimator_var = var_estimator_var.flatten()
    return exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var, x, exact, run_dir

def experiment_1_output_mean_and_variance(fidelities = [2,4,8,16,32], levels = [9,7,5,3,1]):
    exp_estimator_exp, _, var_estimator_exp, _, x, exact, run_dir = initialise(fidelities, levels)
    experiment_subdir = prepare_experiment_directory(run_dir, fidelities, levels, experiment_number = 1)
    plt.figure(figsize=(8,5))
    plt.plot(x , exp_estimator_exp, label=r'$\mathcal{Y}(x)]$.', color = 'orange')
    plt.plot(x, exact, label='Exact Solution', linestyle='dashed', color = 'black')
    plt.fill_between(x, exp_estimator_exp + np.sqrt(var_estimator_exp), exp_estimator_exp - np.sqrt(var_estimator_exp), color='orange', alpha=0.5, label=r'$\mathcal{Y}(x) \pm \sqrt{\mathcal{V}(x)}$.')
    plt.fill_between(x, exp_estimator_exp + 2*np.sqrt(var_estimator_exp), exp_estimator_exp - 2*np.sqrt(var_estimator_exp), color='orange', alpha=0.1, label=r'$\mathcal{Y}(x) \pm 2\sqrt{\mathcal{V}(x)}$.')
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    hash_input = json.dumps({"fidelities": fidelities, "levels": levels}, sort_keys=True)
    unique_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    save_path = os.path.join(experiment_subdir, f"confidence_intervals_eval_{unique_hash}.pdf")
    plt.savefig(save_path)
    plt.close()

def experiment_2_exp_estimator_var_vs_fid(
    fidelities = [[int(v)] for v in np.unique(np.logspace(1, 4, 20, dtype = int)) ],
    level = 10):
    results = []
    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    experiment_subdir = prepare_experiment_directory(run_dir, fidelities, [level], experiment_number = 2)
    npz_file = os.path.join(experiment_subdir, "results.npz")
    if os.path.exists(npz_file):
        print(f"Results already exist. Loading from: {npz_file}")
        data = np.load(npz_file)
        fidelities_flat = data["fidelities"]
        results_array = data["mean_exp_estimator_var"]
    else:
        for f in fidelities:
            _, exp_estimator_var, _, _ = get_estimator(model, x, f, [level])
            results.append(np.mean(exp_estimator_var))
            print('completed fidelity ', f)

        fidelities_flat = np.array([f[0] for f in fidelities])
        results_array = np.array(results)
        npz_file = os.path.join(experiment_subdir, "results.npz")
        np.savez(npz_file, fidelities=fidelities_flat, mean_exp_estimator_var=results_array)

        print(f"Experiment 2 results saved to: {npz_file}")
    log_x = np.log(fidelities_flat)
    log_y = np.log(results_array)

    # Linear regression on log-log data
    slope, intercept, _, _, std_err = linregress(x=log_x, y=log_y)
    result = linregress(x=log_x, y=log_y)
    intercept_stderr = result.intercept_stderr
    # Compute 99% confidence interval for the slope
    n = len(log_x)
    alpha = 0.01
    tval = t.ppf(1 - alpha/2, df=n - 2)
    margin = tval * std_err
    lower = slope - margin
    upper = slope + margin
    print([lower, slope, upper])
    ci_intercept = tval * intercept_stderr
    lower_intercept = intercept - ci_intercept
    upper_intercept = intercept + ci_intercept

    fit = np.exp(slope * log_x + intercept)
    fit_lower = np.exp(lower * log_x + lower_intercept)
    fit_upper = np.exp(upper * log_x + upper_intercept)
    plt.loglog(fidelities_flat, results_array, color='blue', label='Data')
    plt.loglog(fidelities_flat, fit, color='black', linestyle='dashed', label='Fit')
    plt.fill_between(fidelities_flat, fit_lower, fit_upper, color='gray', alpha=0.3, label='99% CI')
    plt.ylabel(r'$\int S_Y^2(x)~ dx$')
    plt.xlabel('T')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(experiment_subdir, f"exp_estimator_var_v_fid.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return None

def experiment_3_var_estimator_var_vs_fid(
    fidelities = [[int(v)] for v in np.unique(np.logspace(1, 4, 20, dtype = int)) ],
    level = 10):
    results = []
    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    experiment_subdir = prepare_experiment_directory(run_dir, fidelities, [level], experiment_number = 3)
    npz_file = os.path.join(experiment_subdir, "results.npz")
    if os.path.exists(npz_file):
        print(f"Results already exist. Loading from: {npz_file}")
        data = np.load(npz_file)
        fidelities_flat = data["fidelities"]
        results_array = data["mean_var_estimator_var"]
    else:
        for f in fidelities:
            _, _, _, var_estimator_var = get_estimator(model, x, f, [level])
            results.append(np.mean(var_estimator_var))
            print('completed fidelity ', f)

        fidelities_flat = np.array([f[0] for f in fidelities])
        results_array = np.array(results)
        npz_file = os.path.join(experiment_subdir, "results.npz")
        np.savez(npz_file, fidelities=fidelities_flat, mean_var_estimator_var=results_array)

        print(f"Experiment 3 results saved to: {npz_file}")
    log_x = np.log(fidelities_flat)
    log_y = np.log(results_array)

    # Linear regression on log-log data
    slope, intercept, _, _, std_err = linregress(x=log_x, y=log_y)
    result = linregress(x=log_x, y=log_y)
    intercept_stderr = result.intercept_stderr
    # Compute 99% confidence interval for the slope
    n = len(log_x)
    alpha = 0.01
    tval = t.ppf(1 - alpha/2, df=n - 2)
    margin = tval * std_err
    lower = slope - margin
    upper = slope + margin
    print([lower, slope, upper])
    ci_intercept = tval * intercept_stderr
    lower_intercept = intercept - ci_intercept
    upper_intercept = intercept + ci_intercept

    fit = np.exp(slope * log_x + intercept)
    fit_lower = np.exp(lower * log_x + lower_intercept)
    fit_upper = np.exp(upper * log_x + upper_intercept)
    plt.loglog(fidelities_flat, results_array, color='blue', label='Data')
    plt.loglog(fidelities_flat, fit, color='black', linestyle='dashed', label='Fit')
    plt.fill_between(fidelities_flat, fit_lower, fit_upper, color='gray', alpha=0.3, label='99% CI')
    plt.ylabel(r'$\int S_V^2(x)~ dx$')
    plt.xlabel('T')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(experiment_subdir, f"var_estimator_var_v_fid.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return None

def set_dropout_probability(model, new_p):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_p

def experiment_4_variance_v_dropout_prob(
    p_vec      = np.linspace(0,1, num = 50, endpoint = False)[1:],
    fidelities = [10000],
    levels     = [10]
):

    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    experiment_subdir = prepare_experiment_directory(run_dir, sum([fidelities, levels],[]), list(p_vec), experiment_number = 4)
    npz_file = os.path.join(experiment_subdir, "results.npz")

    if os.path.exists(npz_file):
        print(f"Results already exist. Loading from: {npz_file}")
        data = np.load(npz_file)
        p_vec   = data["p_vec"]
        exp_v_p = data["exp_v_p"]
        var_v_p = data["var_v_p"]
    else:
        exp_v_p = []
        var_v_p = []
        for p in p_vec:
            set_dropout_probability(model, p)
            _, exp_estimator_var, _, var_estimator_var = get_estimator(model, x, fidelities, levels)
            exp_v_p.append(np.mean(exp_estimator_var))
            var_v_p.append(np.mean(var_estimator_var))
            print('completed probability ', p)
        np.savez(npz_file, p_vec=p_vec, exp_v_p=np.array(exp_v_p), var_v_p=np.array(var_v_p) )
    p = np.array(p_vec)
    def linfit(p, a, b, c, d):
        return a + np.abs(b)*np.abs(d)*np.log(p) - np.abs(c)*np.log(1 - p**np.abs(d))
    params, cov = curve_fit(linfit, p, np.log(np.array(exp_v_p)))
    a, b, c, d = params
    se = np.sqrt(np.diag(cov))

    # 95% CI using t-distribution
    dof = len(exp_v_p) - len(params)  # degrees of freedom
    alpha = 0.01
    tval = t.ppf(1 - alpha/2, dof)

    ci_lower = params - tval * se
    ci_upper = params + tval * se

    print("Parameter estimates expectation:")
    print(f"a = {a:.4f}")
    print(f"b = {b:.4f}")
    print(f"c = {c:.4f}")
    print(f"d = {d:.4f}")

    print("\n95% confidence intervals:")
    print(f"a: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
    print(f"b: [{np.abs(ci_lower[1]):.4f}, {np.abs(ci_upper[1]):.4f}]")
    print(f"c: [{np.abs(ci_lower[2]):.4f}, {np.abs(ci_upper[2]):.4f}]")
    print(f"d: [{np.abs(ci_lower[3]):.4f}, {np.abs(ci_upper[3]):.4f}]")

    fit = np.exp(linfit(p, a, b, c, d))
    all_combos = np.stack([
    np.exp(linfit(p, *params))
    for params in itertools.product(*zip(ci_lower, ci_upper))])
    fit_low  = all_combos.min(axis=0)
    fit_high = all_combos.max(axis=0)

    plt.semilogy(p, np.array(exp_v_p), color='blue', label='Data')
    plt.semilogy(p, fit, color='black', linestyle='dashed', label='Fit')
    plt.fill_between(p, fit_low, fit_high, color='gray', alpha=0.3, label='99% CI')
    plt.ylabel(r'$\int S_Y^2(x)~ dx$')
    plt.xlabel(r'$p_{\rm{drop}}$')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(experiment_subdir, f"exp_estimator_var_v_pdrop.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    params, cov = curve_fit(linfit, p, np.log(np.array(var_v_p)))
    a, b, c, d = params
    se = np.sqrt(np.diag(cov))
    ci_lower = params - tval * se
    ci_upper = params + tval * se

    print("Parameter estimates expectation:")
    print(f"a = {a:.4f}")
    print(f"b = {np.abs(b):.4f}")
    print(f"c = {np.abs(c):.4f}")
    print(f"d = {np.abs(d):.4f}")

    print("\n99% confidence intervals:")
    print(f"a: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
    print(f"b: [{np.abs(ci_lower[1]):.4f}, {np.abs(ci_upper[1]):.4f}]")
    print(f"c: [{np.abs(ci_lower[2]):.4f}, {np.abs(ci_upper[2]):.4f}]")
    print(f"d: [{np.abs(ci_lower[3]):.4f}, {np.abs(ci_upper[3]):.4f}]")

    fit = np.exp(linfit(p, a, b, c, d))
    all_combos = np.stack([
    np.exp(linfit(p, *params))
    for params in itertools.product(*zip(ci_lower, ci_upper))])
    fit_low  = all_combos.min(axis=0)
    fit_high = all_combos.max(axis=0)

    plt.semilogy(p, np.array(var_v_p), color='blue', label='Data')
    plt.semilogy(p, fit, color='black', linestyle='dashed', label='Fit')
    plt.fill_between(p, fit_low, fit_high, color='gray', alpha=0.3, label='99% CI')
    plt.ylabel(r'$\int S_V^2(x)~ dx$')
    plt.xlabel(r'$p_{\rm{drop}}$')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(experiment_subdir, f"var_estimator_var_v_pdrop.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return None

def all_levels(cost=1000, T=[4,8,16]):
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

def experiment_5_var_v_levels_fixed_cost(c = 1000, T = [4,8,16]):
    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    experiment_subdir = prepare_experiment_directory(run_dir, [c], T, experiment_number = 5)
    npz_file = os.path.join(experiment_subdir, "results.npz")

    if os.path.exists(npz_file):
        print(f"Results already exist. Loading from: {npz_file}")
        data = np.load(npz_file)
        M_all   = data["M_all"]
        exp_var = data["exp_var"]
        var_var = data["var_var"]
    else:
        M_all = all_levels(cost = c, T=T)
        exp_var = []
        var_var = []
        start_time = time.time()
        for i, M in enumerate(M_all):
            iter_start = time.time()
            _, exp_estimator_var, _, var_estimator_var = get_estimator(model, x, T, M)
            exp_var.append(np.mean(exp_estimator_var))
            var_var.append(np.mean(var_estimator_var))

            elapsed = time.time() - start_time
            iters_done = i + 1
            total_iters = len(M_all)
            avg_per_iter = elapsed / iters_done
            remaining = avg_per_iter * (total_iters - iters_done)

            print(
                f"Completed {iters_done} out of {total_iters} "
                f"— elapsed: {elapsed:.1f}s, "
                f"ETA: {remaining:.1f}s"
            )
        np.savez(npz_file, M_all = np.array(M_all), exp_var = np.array(exp_var), var_var = np.array(var_var))

    x = np.array([point[1] for point in M_all])
    y = np.array([point[2] for point in M_all])
    alpha = 0.005
    N = int(alpha*len(exp_var))
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
    save_path = os.path.join(experiment_subdir, f"var_exp_estimator_fixed_cost.pdf")
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
    save_path = os.path.join(experiment_subdir, f"var_var_estimator_fixed_cost.pdf")
    plt.scatter(avg_min_M1, avg_min_M2, c='k', marker = 'x')
    plt.scatter(theo[1], theo[2], c='r', marker = 'x')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def compute_theoretical_M_exp(T=[4, 8, 16], c_T=1000):
    T = np.array(T)
    L = len(T) - 1

    diffs = T[1:] - T[:-1]
    denom_terms = diffs / np.sqrt(T[1:] * T[:-1])
    denominator = 1 + np.sum(denom_terms)


    M0 = c_T / T[0] / denominator
    M_rest = c_T / np.sqrt(T[1:] * T[:-1]) / denominator
    M = np.concatenate(([M0], M_rest))
    return M

def compute_theoretical_M_var(T = [4, 8, 16], c_T = 1000):
    T = np.array(T)
    L = len(T) - 1
    base_term = np.sqrt(T[0] / (T[0] - 1))

    diffs = T[1:] - T[:-1]
    denom_terms = diffs / np.sqrt((T[1:] - 1) * (T[:-1] - 1))

    denominator = base_term + np.sum(denom_terms)

    M0 = c_T / np.sqrt(T[0] * (T[0] - 1)) / denominator
    M_rest = c_T / np.sqrt((T[1:] - 1) * (T[:-1] - 1)) / denominator

    return np.concatenate(([M0], M_rest))

def experiment_6_dropout_bias(
    num_drop   = 50,
    fidelities = [10000],
    levels     = [10]):
    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    experiment_subdir = prepare_experiment_directory(run_dir, fidelities+levels , [num_drop], experiment_number = 6)
    npz_file = os.path.join(experiment_subdir, "results.npz")
    exact = exact_solution(CONFIG, device = "cpu")
    exact = exact.numpy()

    if os.path.exists(npz_file):
        print(f"Results already exist. Loading from: {npz_file}")
        data = np.load(npz_file)
        p_vec   = data["p_vec"]
        exp_v_p = data["exp_v_p"]
        var_v_p = data["var_v_p"]

    else:
        model.eval()
        u0 = model(x).detach().numpy()
        exp_v_p = [np.sqrt(np.mean((u0 - exact)**2))]
        var_v_p = [0.0]
        model.train()
        p_vec  = np.linspace(0,1,num_drop+1, endpoint= False)[1:]
        for p in p_vec:
            set_dropout_probability(model, p)
            exp_estimator_exp, _, var_estimator_exp, _ = get_estimator(model, x, fidelities, levels)
            exp_v_p.append(np.sqrt(np.mean((exp_estimator_exp - exact)**2)))
            var_v_p.append(np.sqrt(np.mean(var_estimator_exp**2)))
            print('completed probability ', p)
        npz_file = os.path.join(experiment_subdir, "results.npz")
        p_vec = np.linspace(0,1,num_drop+1, endpoint= False)
        np.savez(npz_file, exp_v_p=exp_v_p, var_v_p=var_v_p, p_vec = p_vec)
        print(f"Experiment 2 results saved to: {npz_file}")

    plt.plot(p_vec[1:], exp_v_p[1:])
    plt.show()
    plt.close()
    plt.semilogy(p_vec, var_v_p)
    plt.show()

if __name__ == "__main__":
    # experiment_1_output_mean_and_variance(fidelities = [10], levels = [1])
    # experiment_1_output_mean_and_variance(fidelities = [100], levels = [1])
    # experiment_1_output_mean_and_variance(fidelities = [1000], levels = [1])
    # experiment_1_output_mean_and_variance(fidelities = [10000], levels = [1])
    # experiment_4_variance_v_dropout_prob()
    # experiment_5_var_v_levels_fixed_cost()
    experiment_6_dropout_bias()
