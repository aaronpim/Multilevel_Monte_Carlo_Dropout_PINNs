import torch
import numpy as np
from Load_Model import load_latest_model

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

def output_exp_var_computation(model, x, data = [], fid = 1000, u_or_f = 'f'):
    if fid > len(data):
        for _ in range(fid - len(data)):
            u, f = model(x)
            if u_or_f == 'f':
                data.append(f.detach().numpy())
            else:
                data.append(u.detach().numpy())
    Y = np.mean(np.stack(data, axis=0), axis=0)
    V = np.var(np.stack(data, axis=0), axis=0)
    return Y, V, data

def ML_evals(model, x, fidelities = [2,4,8,16,32], levels = [9,7,5,3,1], u_or_f = 'f'):
    fidelities = validate_list(fidelities)
    levels     = validate_list(levels)
    if not len(fidelities) == len(levels):
        return None
    Y_vec = [[] for _ in range(len(fidelities))]
    V_vec = [[] for _ in range(len(fidelities))]
    while sum(levels) > 0:
        data = []
        for i,f in enumerate(fidelities):
            if levels[i] > 0:
                Y, V, data = output_exp_var_computation(model, x, data=data, fid = f, u_or_f = u_or_f)
                Y_vec[i].append(Y)
                V_vec[i].append(V)
                levels[i] += -1
    Y_vec = [np.stack(Y_i, axis=0).squeeze() for Y_i in Y_vec]
    V_vec = [np.stack(V_i, axis=0).squeeze() for V_i in V_vec]
    return Y_vec, V_vec

def X_diff(vec):
    diff_vec = [[] for _ in range(len(vec)-1)]
    for i in range(1,len(vec)):
        n   = vec[i]
        nm1 = vec[i-1]
        N = min([len(n), len(nm1)])
        for j in range(N):
            diff_vec[i-1].append(n[j]-nm1[j])
    diff_vec = [np.stack(dV, axis=0).squeeze() for dV in diff_vec]
    return diff_vec

def get_estimator_exp_var(vec):
    if len(vec) == 1 and len(vec[0].shape) == 1:
        return vec[0], np.array([])
    exp = np.mean(vec[0], axis=0)
    var = np.var(vec[0], axis=0, ddof=1)/vec[0].shape[0]
    if len(vec) > 1:
        diff_vec = X_diff(vec)
        for dV in diff_vec:
            exp += np.mean(dV, axis=0)
            var += np.var(dV, axis=0, ddof=1)/dV.shape[0]
    return exp, var

def get_estimator(model = None, x = None, fidelities = [2,4,8,16,32], levels = [9,7,5,3,1], u_or_f = 'f'):
    if model is None:
        model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    Y_vec, V_vec = ML_evals(model, x, fidelities = fidelities, levels = levels, u_or_f = u_or_f)
    exp_estimator_exp, exp_estimator_var = get_estimator_exp_var(Y_vec)
    var_estimator_exp, var_estimator_var = get_estimator_exp_var(V_vec)
    return exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var

if __name__ == "__main__":
    model, x, CONFIG, run_dir = load_latest_model(device = "cpu")
    exp_estimator_exp, exp_estimator_var, var_estimator_exp, var_estimator_var = get_estimator(model, x, fidelities = [32,64], levels = [10,5], u_or_f = 'u')
    print(exp_estimator_exp.shape)
    print(var_estimator_exp.shape)
    print(exp_estimator_var.shape)
    print(var_estimator_var.shape)
