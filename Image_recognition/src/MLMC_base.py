import os
import torch
from src.trainmodel import set_seed, CONFIG_to_folder_path, load_all
'''
The objective of this code is to take the stochastic neural network, called model, and
estimate the mean and variance of the mean esimtator Y, and the mean and variance of
the variance esimtator V, according to MLMC principles.
'''
def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 128,
        "drop_p": 0.05,
        "num_conv_layers": 3,
        "width" : 64,
        "MLP_width": 128,
        "kernal_size": 3,
        "padding": 1
        }
    CONFIG.update(overwrites)
    return CONFIG

def compute_mean_and_var(evals):
    '''

    this function takes the evals and computes the max and variance estimator.
    Using the uniform grid over the domain with the integrating factor.
    '''
    max_est = evals.mean(dim = 0).mean()
    var_est  = evals.var(dim = 0).mean()
    return max_est.item(), var_est.item()

def single_level_single_fidelity_eval(model, images, labels, num_evals = None, prev_evals = None):
    """
    The model produces raw logits with shape [num_images, num_classes].
    The model is evaluated ``num_evals`` times with Monte Carlo dropout enabled,
    producing an array of shape [num_evals, num_images, num_classes].

    A softmax is then applied along the class dimension to convert the logits
    into class probabilities. For each image, only the probability assigned to
    its true class label is retained, resulting in an array of shape
    [num_evals, num_images]. This scalar quantity is treated as the Monte Carlo
    output for subsequent estimation of the sample mean and sample variance.

    The optional ``prev_evals`` argument allows previously computed evaluations
    to be reused, avoiding unnecessary forward passes. Although it is not
    currently required by the present implementation, it provides useful
    flexibility for future extensions.
    """
    if num_evals is None:
        num_evals = 16
    with torch.no_grad():
        evals = torch.stack([model(images).squeeze() for _ in range(num_evals)])
        probs   = torch.softmax(evals, dim = -1)
        output = probs[:,torch.arange(probs.shape[1]), labels]
        if prev_evals is not None:
            output = torch.cat([output, prev_evals])
        return output

def single_level_multi_fidelity_eval(model, images, labels, fidelity_ladder = None):
    '''
    For a given fidelity ladder this function computes the coarsest estimation of the mean and variance
    denoted Y0 and V0 respectively, and also computes the vector of correctors dY and dV. This implicitly
    assumes that the number of evaluations per fidelity is [1, 1, ... , 1].
    '''
    if fidelity_ladder is None:
        fidelity_ladder = [2, 4, 8, 16, 32, 64, 128]
    max_fid = max(fidelity_ladder)
    evals = single_level_single_fidelity_eval(model, images, labels, num_evals = max_fid)
    running_max  = []
    running_var  = []
    for fid in fidelity_ladder:
        max_est, var_est = compute_mean_and_var(evals[:fid])
        running_max.append(max_est)
        running_var.append(var_est)
    running_max = torch.tensor(running_max)
    running_var  = torch.tensor(running_var)
    max_est_0 = running_max[0]
    var_est_0 = running_var[0]
    d_max_est = torch.diff(running_max)
    d_var_est = torch.diff(running_var)
    return max_est_0, d_max_est, var_est_0, d_var_est

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

def multi_level_multi_fidelity_eval(model, images, labels, fidelity_ladder = None, num_levels = None):
    '''
    Performs a complete multilevel, multifidelity Monte Carlo (MLMC)
    evaluation of the stochastic neural network.

    For each fidelity level, the required number of independent samples is
    generated according to the MLMC sampling schedule. The resulting coarse
    estimators and correction terms are accumulated before computing:

        - the expected value of the max estimator,
        - the variance of the max estimator,
        - the expected value of the variance estimator,
        - the variance of the variance estimator.

    The sampling strategy assumes one model evaluation is added each time
    the fidelity increases along the fidelity ladder.
    '''
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
                mean_est_0, d_mean_est, var_est_0, d_var_est  = single_level_multi_fidelity_eval(model, images, labels, fidelity_ladder = fidelity_ladder[:(i+1)])
                Y_store, V_store = store_vals(Y_store, V_store, mean_est_0, d_mean_est, var_est_0, d_var_est)
    mean_estimate = expectation_of_estimator(Y_store)
    mean_estimator_variance = variance_of_estimator(Y_store, num_levels)
    sample_variance_estimate = expectation_of_estimator(V_store)
    sample_variance_estimator_variance = variance_of_estimator(V_store, num_levels)
    return mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance


if __name__ == "__main__":
    set_seed(seed = 0)
    CONFIG = make_config(seed = 0)
    model, _, hold_loader  = load_all(CONFIG, device = 'cpu')
    model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
    model.train()
    with torch.no_grad():
        images = []
        labels = []
        for image, label in hold_loader:
            images.append(image)
            labels.append(label)
        images = torch.cat(images, dim = 0)[:1000]
        labels = torch.cat(labels, dim = 0)[:1000]
        mean_estimate, mean_estimator_variance, sample_variance_estimate, sample_variance_estimator_variance = multi_level_multi_fidelity_eval(model, images, labels, fidelity_ladder = [2,4], num_levels = [5,5])
        print(mean_estimate.item(), mean_estimator_variance.item(), sample_variance_estimate.item(), sample_variance_estimator_variance.item())
