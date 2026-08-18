import os
import torch
from src.trainmodel import load_all, compute_delaunay_weights, set_seed
'''
The objective of this code is to take the stochastic neural network, called model, and
estimate the mean and variance of the mean esimtator Y, and the mean and variance of
the variance esimtator V, according to MLMC principles.
'''
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

def compute_mean_and_var(evals, factor = None, weights = None):
    '''
    The raw evaluations of the model are shaped like [number of input points, Nx, Ny, Nz],
    this function takes the evals and computes the integrated mean and variance estimator.
    Using the uniform grid over the spatial domain with the integrating factor. We then
    use delaunay weights to integrate over the input points.
    '''
    if factor is None:
        factor =  (8/27)
    if weights is None:
        CONFIG = make_config(seed = 0)
        _, values, _, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
        weights = compute_delaunay_weights(values, device = 'cpu')
    mean_est = ((factor*evals.mean(dim = 0).sum(dim = [-1,-2,-3]))*weights).sum()
    var_est = ((factor*evals.var(dim = 0).sum(dim = [-1,-2,-3]))*weights).sum()
    return mean_est.item(), var_est.item()

def single_level_single_fidelity_eval(model, values, num_evals = None, prev_evals = None):
    '''
    For a single level and fidelity this function evaluates the model num_evals times. It
    has the option for using previous terms with prev_evals but that is unlikely given how
    the next function is formulated, but it is a useful feature to have regardless.
    '''
    if num_evals is None:
        num_evals = 16
    with torch.no_grad():
        evals = torch.stack([model(values) for _ in range(num_evals)])
        if prev_evals is not None:
            evals = torch.cat([evals, prev_evals])
        return evals

def single_level_multi_fidelity_eval(model, values, fidelity_ladder = None, factor = None, weights = None):
    '''
    For a given fidelity ladder this function computes the coarsest estimation of the mean and variance
    denoted Y0 and V0 respectively, and also computes the vector of correctors dY and dV. This implicitly
    assumes that the number of evaluations per fidelity is [1, 1, ... , 1].
    '''
    if factor is None:
        factor =  (8/27)
    if fidelity_ladder is None:
        fidelity_ladder = [2, 4, 8, 16, 32, 64, 128]
    if weights is None:
        weights = compute_delaunay_weights(values, device = 'cpu')
    max_fid = max(fidelity_ladder)
    evals = single_level_single_fidelity_eval(model, values, num_evals = max_fid)
    running_mean = []
    running_var  = []
    for fid in fidelity_ladder:
        mean_est, var_est = compute_mean_and_var(evals[:fid], factor = factor, weights = weights)
        running_mean.append(mean_est)
        running_var.append(var_est)
    running_mean = torch.tensor(running_mean)
    running_var  = torch.tensor(running_var)
    mean_est_0 = running_mean[0]
    var_est_0 = running_var[0]
    d_mean_est = torch.diff(running_mean)
    d_var_est = torch.diff(running_var)
    return mean_est_0, d_mean_est, var_est_0, d_var_est

def levels_to_num_evals(num_levels = None):
    '''
    Given a list containing the number of evaluations per level M = [M0, ... ML]
    This function returns the number of passes of single_level_multi_fidelity_eval
    are needed to meet these demands. This implicitly assumes that the number of
    evaluations per fidelity is of the form [1,... ,1,0 ,... ,0].
    '''
    if num_levels is None:
        num_levels = [15, 12, 10, 8, 6, 4, 2]
    num_levels = torch.tensor(num_levels)
    return torch.cat([ num_levels[:-1] - num_levels[1:], num_levels[-1:] ])

def store_vals(Y_store, V_store, Y0, dY, V0, dV):
    '''
    Stores one realisation of the MLMC estimators.

    The first entry contains the coarse estimator (Y0 or V0), while the
    remaining entries contain the correction terms (dY or dV) associated
    with each fidelity level.
    '''
    Y_store[0].append(Y0)
    V_store[0].append(V0)
    for j in range(len(dY)):
        Y_store[j+1].append(dY[j])
        V_store[j+1].append(dV[j])
    return Y_store, V_store

def expectation_of_estimator(store):
    '''
    Computes the MLMC estimate of the expectation.

    Each entry of 'store' contains independent samples from one MLMC level.
    The estimator is obtained by averaging each level separately and then
    summing the resulting level averages.
    '''
    expectation = 0
    for vec in store:
        expectation += torch.tensor(vec).mean()
    return expectation

def variance_of_estimator(store, levels):
    '''
    Computes the variance of the MLMC estimator.

    Since each MLMC level is sampled independently, the estimator variance
    is the sum of the sample variances divided by the number of samples
    taken on each level.
    '''
    variance = 0
    for i, vec in enumerate(store):
        variance += torch.tensor(vec).var()/levels[i]
    return variance

def multi_level_multi_fidelity_eval(model, values, fidelity_ladder = None, num_levels = None, factor = None, weights = None):
    '''
    Performs a complete multilevel, multifidelity Monte Carlo (MLMC)
    evaluation of the stochastic neural network.

    For each fidelity level, the required number of independent samples is
    generated according to the MLMC sampling schedule. The resulting coarse
    estimators and correction terms are accumulated before computing:

        - the expected value of the mean estimator,
        - the variance of the mean estimator,
        - the expected value of the variance estimator,
        - the variance of the variance estimator.

    The sampling strategy assumes one model evaluation is added each time
    the fidelity increases along the fidelity ladder.
    '''
    if factor is None:
        factor =  (8/27)
    if fidelity_ladder is None:
        fidelity_ladder = [2, 4, 8, 16, 32, 64]
    if num_levels is None:
        num_levels = [15, 12, 10, 8, 6, 4]
    if weights is None:
        weights = compute_delaunay_weights(values, device = 'cpu')
    Y_store = [[] for _ in range(len(num_levels))]
    V_store = [[] for _ in range(len(num_levels))]
    num_evals_per_fid = levels_to_num_evals(num_levels)
    for i, num_evals_for_fid_f in enumerate(num_evals_per_fid):
        if num_evals_for_fid_f == 0:
            continue
        else:
            for _ in range(num_evals_for_fid_f):
                mean_est_0, d_mean_est, var_est_0, d_var_est  = single_level_multi_fidelity_eval(model, values, fidelity_ladder = fidelity_ladder[:(i+1)], factor = factor, weights = weights)
                Y_store, V_store = store_vals(Y_store, V_store, mean_est_0, d_mean_est, var_est_0, d_var_est)
    mean_estimate = expectation_of_estimator(Y_store)
    mean_estimator_variance = variance_of_estimator(Y_store, num_levels)
    sample_variance_estimate = expectation_of_estimator(V_store)
    sample_variance_estimator_variance = variance_of_estimator(V_store, num_levels)
    return mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance


if __name__ == "__main__":
    set_seed(seed = 0)
    CONFIG = make_config(seed = 0)
    model, values, _, _, _, _, _, _ = load_all(CONFIG, device = 'cpu')
    weights = compute_delaunay_weights(values, device = 'cpu')
    mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance = multi_level_multi_fidelity_eval(model, values, fidelity_ladder = [2, 4, 8], num_levels = [6, 4, 2])
    print(mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance)
