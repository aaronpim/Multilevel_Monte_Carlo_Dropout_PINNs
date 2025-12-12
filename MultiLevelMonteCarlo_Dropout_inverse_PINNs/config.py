CONFIG = {
    "layers": [1, 128, 128, 128, 128],
    "dropout_rate": 0.2,
    "seed": 0,
    "num_x_points": 200,
    "random_eps_width": 0.025,
    "alpha":1e-4,
    "beta":1e-4,
    "uzawa_coeff": 2.5e-5,
    "epochs": 200000,
    "uzawa_epoch": 50,
    "learning_rate": 1e-3,
    "global_or_local": 'local',
    "num_dropout_repeats": 5,
    "lagrange_multiplier_reps": 20
}
