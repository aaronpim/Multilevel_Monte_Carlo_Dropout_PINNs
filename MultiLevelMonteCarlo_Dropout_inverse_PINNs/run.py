import os
import sys
from config import CONFIG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from model_defn_and_training import train, set_seed
from MLMC_experiments import experiment_1_output_mean_and_variance, experiment_2_exp_estimator_var_vs_fid, experiment_3_var_estimator_var_vs_fid, experiment_4_variance_v_dropout_prob, experiment_5_var_v_levels_fixed_cost

if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    #train(CONFIG)
    # experiment_1_output_mean_and_variance(fidelities = [10], levels = [1])
    # experiment_1_output_mean_and_variance(fidelities = [100], levels = [1])
    # experiment_1_output_mean_and_variance(fidelities = [1000], levels = [1])
    # experiment_1_output_mean_and_variance(fidelities = [10000], levels = [1])
    #
    # experiment_1_output_mean_and_variance(fidelities = [10], levels = [1], u_or_f = 'u')
    # experiment_1_output_mean_and_variance(fidelities = [100], levels = [1], u_or_f = 'u')
    # experiment_1_output_mean_and_variance(fidelities = [1000], levels = [1], u_or_f = 'u')
    # experiment_1_output_mean_and_variance(fidelities = [10000], levels = [1], u_or_f = 'u')

    # experiment_2_exp_estimator_var_vs_fid(u_or_f = 'f')
    # experiment_2_exp_estimator_var_vs_fid(u_or_f = 'u')
    #
    # experiment_3_var_estimator_var_vs_fid(u_or_f = 'f')
    # experiment_3_var_estimator_var_vs_fid(u_or_f = 'u')
    #
    # experiment_4_variance_v_dropout_prob(u_or_f = 'f')
    # experiment_4_variance_v_dropout_prob(u_or_f = 'u')

    experiment_5_var_v_levels_fixed_cost(u_or_f = 'f')
    experiment_5_var_v_levels_fixed_cost(u_or_f = 'u')
