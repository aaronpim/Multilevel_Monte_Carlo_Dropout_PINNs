import os
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.loss_defn import random_disk_points
from src.trainmodel import set_seed, CONFIG_to_folder_path
from src.model_defn import load_model
from src.MLMC_base import make_config

def get_model(seed = 0):
    set_seed(seed = seed)
    CONFIG = make_config(seed = 0)
    model  = load_model(CONFIG, device = 'cpu')
    model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
    model.train()
    return model

def evaluate_model(model = None, num_evals = 16, num_pts = 512):
    if model is None:
        model = get_model()
    output = []
    for _ in range(num_evals):
        input_pairs = random_disk_points(num_pts, 'cpu')
        output.append(model(input_pairs).squeeze())
    return torch.stack(output)

def g(evals):
    mean = evals.mean(dim=0)
    second = (evals**2).mean(dim=0)
    return mean, second

def diagnostic(mean, second):
    return second - mean**2

def coupled_coarse_estimators(evals = None, N = None):
    if evals is None:
        evals =  evaluate_model()
    if N is None:
        N = evals.shape[0]//2

    evals_a = evals[:N]
    evals_b = evals[N:]
    meanA, m2A = g(evals_a)
    meanB, m2B = g(evals_b)
    meanF = 0.5*(meanA + meanB)
    m2F   = 0.5*(m2A + m2B)

    PhiA = diagnostic(meanA, m2A)
    PhiB = diagnostic(meanB, m2B)
    PhiF = diagnostic(meanF, m2F)
    Delta = PhiF - 0.5*PhiA - 0.5*PhiB
    return Delta

def main_experiment(fid_list = [8, 16, 32, 64, 128, 256, 512], num_reps = 100, model_seed = 0):
    output_vec = []
    start_time = time.time()
    count = 0
    completed_work = 0
    total_work = sum(fid * num_reps for fid in fid_list)
    with torch.no_grad():
        model = get_model(seed = model_seed)
        for fid in fid_list:
            temp_vec = []
            for _ in range(num_reps):
                evals = evaluate_model(model, num_evals = fid)
                Delta = coupled_coarse_estimators(evals)
                temp_vec.append(Delta)
                count += 1
                completed_work += fid

                elapsed = time.time() - start_time
                work_rate = completed_work / elapsed
                remaining_work = total_work - completed_work
                eta = remaining_work / work_rate / 60

                print(
                    f'Completed {count}/{len(fid_list)*num_reps}. '
                    f'ETA: {eta:.2f} minutes'
                )
            temp_vec = torch.stack(temp_vec)
            results  = temp_vec.var()
            output_vec.append([fid, results])
        output_vec = torch.tensor(output_vec)
        torch.save(output_vec, f'plots/model_diff_evals.pt')

if __name__ == "__main__":
    # main_experiment()
    output_vec = torch.load(f'plots/model_diff_evals.pt', weights_only = True)
    plt.loglog(output_vec[:,0], output_vec[:,1], color = 'orange', marker  = 'o')
    plt.xscale('log', base=2)
    plt.ylabel(r'$\rm{Var}[\Delta_{\ell}]$')
    plt.xlabel(r'$T_{\ell}$')
    plt.tight_layout()
    plt.savefig(f'plots/model_diff_evals.pdf')
    plt.close()
    from scipy.stats import linregress, t
    logx = np.log2(output_vec[:,0].numpy())
    logy = np.log2(output_vec[:,1].numpy())
    res = linregress(logx, logy)
    slope = res.slope
    n = len(logx)
    alpha = 0.05
    tval = t.ppf(1 - alpha/2, df=n-2)
    ci_low = slope - tval * res.stderr
    ci_high = slope + tval * res.stderr
    print(slope, ci_low, ci_high)
