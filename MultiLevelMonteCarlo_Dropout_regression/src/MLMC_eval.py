import torch
import numpy as np
import os
from src.trainmodel import CONFIG_to_folder_path, load_all

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
        "dropout_prob": 0.001,
        "epochs": 10000,
        "lr": 1e-4,
        "power": 2,
        }
    CONFIG.update(overwrites)
    return CONFIG

def validate_list(fidelities):
    if not isinstance(fidelities, (list, tuple)):
        raise ValueError("fidelities must be a list or tuple.")
    cleaned = []
    for i, f in enumerate(fidelities):
        if isinstance(f, bool):
            raise ValueError(f"fidelity[{i}] cannot be a boolean.")
        try:
            f_int = int(f)
        except (ValueError, TypeError):
            raise ValueError(f"fidelity[{i}] = {f!r} cannot be converted to an integer.")
        if f_int < 0:
            raise ValueError(f"fidelity[{i}] = {f_int} cannot be negative.")
        cleaned.append(f_int)
    return cleaned

def output_exp_var_computation(model, x, data = [], fid = 1000):
    with torch.no_grad():
        if fid > len(data):
            for _ in range(fid - len(data)):
                data.append(model(x))
        Y = torch.mean(torch.stack(data), dim=0)
        V = torch.var(torch.stack(data), dim=0)
        return Y, V, data

def ML_evals(model, x, fidelities = [2,4,8,16,32], levels = [9,7,5,3,1]):
    fidelities = validate_list(fidelities)
    levels     = validate_list(levels)
    if not len(fidelities) == len(levels):
        return None
    with torch.no_grad():
        Y_vec = [[] for _ in range(len(fidelities))]
        V_vec = [[] for _ in range(len(fidelities))]
        while sum(levels) > 0:
            data = []
            for i,f in enumerate(fidelities):
                if levels[i] > 0:
                    Y, V, data = output_exp_var_computation(model, x, data=data, fid = f)
                    Y_vec[i].append(Y)
                    V_vec[i].append(V)
                    levels[i] += -1
        Y_vec = [torch.stack(Y_i) for Y_i in Y_vec]
        V_vec = [torch.stack(V_i) for V_i in V_vec]
        return Y_vec, V_vec

def X_diff(vec):
    diff_vec = [[] for _ in range(len(vec)-1)]
    for i in range(1,len(vec)):
        n   = vec[i]
        nm1 = vec[i-1]
        N = min([len(n), len(nm1)])
        for j in range(N):
            diff_vec[i-1].append(n[j]-nm1[j])
    diff_vec = [torch.stack(dV) for dV in diff_vec]
    return diff_vec

def get_estimator_exp_var(vec):
    if len(vec) == 1 and len(vec[0].shape) == 1:
        return vec[0], torch.tensor([])
    exp = torch.mean(vec[0], dim=0)
    var = torch.var(vec[0], dim=0)/vec[0].shape[0]
    if len(vec) > 1:
        diff_vec = X_diff(vec)
        for dV in diff_vec:
            if dV.shape[0] > 0:
                exp += torch.mean(dV, dim=0)
            if dV.shape[0] > 1:
                var += torch.var(dV, dim=0)/dV.shape[0]
    return exp, var

def get_estimator(model, x, fidelities = [2,4,8,16,32], levels = [9,7,5,3,1]):
    Y_vec, V_vec = ML_evals(model, x, fidelities = fidelities, levels = levels)
    exp_estimator_exp, exp_estimator_var = get_estimator_exp_var(Y_vec)
    var_estimator_exp, var_estimator_var = get_estimator_exp_var(V_vec)
    return exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var

if __name__ == "__main__":
    CONFIG = make_config(seed = 0)
    model, values, _, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
    values = torch.zeros(1,values.shape[1])
    model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
    model.train()

    exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var = get_estimator(model, values, fidelities = [2,4,8,16,32], levels = [9,7,3,5,1])
    print(exp_estimator_exp.shape)
