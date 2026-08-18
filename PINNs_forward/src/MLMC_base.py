import os
import torch
from src.trainmodel import set_seed, CONFIG_to_folder_path
from src.model_defn import load_PirateNet
from src.loss_defn import gen_x_and_eps
'''
The objective of this code is to take the stochastic neural network, called model, and
estimate the mean and variance of the mean esimtator Y, and the mean and variance of
the variance esimtator V, according to MLMC principles.
'''
def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "input_dim": 2,
        "hidden_dim": 48,
        "num_blocks": 7,
        "output_dim": 1,
        "p_drop": 0.01,
        "activation": "nn.SiLU()",
        "sigma": 1.0,
        "x_num": 101,
        "eps_num": 301,
        "eps_min": -8,
        "eps_max": 4,
        "BC_coef": 1.0,
        "lr": 1e-3,
        "epochs": 5000,
        }
    CONFIG.update(overwrites)
    return CONFIG

def MLMC_config(**overwrites):
    return make_config(x_num = 51, eps_num = 101)

def compute_mean_and_var(evals, factor = None):
    '''
    The raw evaluations of the model are shaped like [number of input points, Nx, Neps],
    this function takes the evals and computes the integrated mean and variance estimator.
    Using the uniform grid over the domain with the integrating factor.
    '''
    if factor is None:
        CONFIG = MLMC_config()
        factor = (CONFIG["eps_max"] - CONFIG["eps_min"])/(CONFIG["x_num"]*CONFIG["eps_num"])
    mean_est = factor*evals.mean(dim = 0).sum(dim = [-1,-2])
    var_est  = factor*evals.var(dim = 0).sum(dim = [-1,-2])
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
        evals = torch.stack([model(values).squeeze() for _ in range(num_evals)])
        if prev_evals is not None:
            evals = torch.cat([evals, prev_evals])
        return evals

def single_level_multi_fidelity_eval(model, values, fidelity_ladder = None, factor = None):
    '''
    For a given fidelity ladder this function computes the coarsest estimation of the mean and variance
    denoted Y0 and V0 respectively, and also computes the vector of correctors dY and dV. This implicitly
    assumes that the number of evaluations per fidelity is [1, 1, ... , 1].
    '''
    if factor is None:
        CONFIG = MLMC_config()
        factor = (CONFIG["eps_max"] - CONFIG["eps_min"])/(CONFIG["x_num"]*CONFIG["eps_num"])
    if fidelity_ladder is None:
        fidelity_ladder = [2, 4, 8, 16, 32, 64, 128]
    max_fid = max(fidelity_ladder)
    evals = single_level_single_fidelity_eval(model, values, num_evals = max_fid)
    running_mean = []
    running_var  = []
    for fid in fidelity_ladder:
        mean_est, var_est = compute_mean_and_var(evals[:fid], factor = factor)
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

def multi_level_multi_fidelity_eval(model, values, fidelity_ladder = None, num_levels = None, factor = None):
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
        CONFIG = MLMC_config()
        factor = (CONFIG["eps_max"] - CONFIG["eps_min"])/(CONFIG["x_num"]*CONFIG["eps_num"])
    if fidelity_ladder is None:
        fidelity_ladder = [2, 4, 8, 16, 32, 64]
    if num_levels is None:
        num_levels = [15, 12, 10, 8, 6, 4]
    Y_store = [[] for _ in range(len(num_levels))]
    V_store = [[] for _ in range(len(num_levels))]
    num_evals_per_fid = levels_to_num_evals(num_levels)
    for i, num_evals_for_fid_f in enumerate(num_evals_per_fid):
        if num_evals_for_fid_f == 0:
            continue
        else:
            for _ in range(num_evals_for_fid_f):
                mean_est_0, d_mean_est, var_est_0, d_var_est  = single_level_multi_fidelity_eval(model, values, fidelity_ladder = fidelity_ladder[:(i+1)])
                Y_store, V_store = store_vals(Y_store, V_store, mean_est_0, d_mean_est, var_est_0, d_var_est)
    mean_estimate = expectation_of_estimator(Y_store)
    mean_estimator_variance = variance_of_estimator(Y_store, num_levels)
    sample_variance_estimate = expectation_of_estimator(V_store)
    sample_variance_estimator_variance = variance_of_estimator(V_store, num_levels)
    return mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance


if __name__ == "__main__":
    print(MLMC_config())
    # set_seed(seed = 0)
    # CONFIG = make_config(seed = 0)
    # model  = load_PirateNet(CONFIG, device = 'cpu')
    # model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
    # model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
    # model.train()
    # values, _, _, _, _ = gen_x_and_eps(CONFIG, device = 'cpu')
    # mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance = multi_level_multi_fidelity_eval(model, values, fidelity_ladder = [2, 4, 8], num_levels = [4, 3, 2])
    # print(mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance)
