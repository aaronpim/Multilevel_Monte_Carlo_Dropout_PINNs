import os
import importlib.util
import torch
from model_defn_and_training import define_x_domain, PINN

# -------------------------------------------------------------------
# Resolve project root (parent of src/)
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

# -------------------------------------------------------------------
# Helper: load python module from path (config.py inside run folder)
# -------------------------------------------------------------------
def load_config_from_path(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.CONFIG


# -------------------------------------------------------------------
# Helper: find newest run folder
# -------------------------------------------------------------------
def get_latest_run(results_root=RESULTS_ROOT):
    if not os.path.exists(results_root):
        raise FileNotFoundError(f"Results directory '{results_root}' does not exist.")

    runs = [
        os.path.join(results_root, d)
        for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)) and d.startswith("run_")
    ]

    if not runs:
        raise FileNotFoundError(f"No run folders found inside '{results_root}'.")

    runs.sort(key=os.path.getmtime)
    latest_run = runs[-1]
    return latest_run


# -------------------------------------------------------------------
# Main loader
# -------------------------------------------------------------------
def load_latest_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get most recent run/ under root/results/
    run_dir = get_latest_run()

    print(f"Loading latest run: {run_dir}")

    # Paths inside the run directory
    config_path = os.path.join(run_dir, "config.py")
    model_path = os.path.join(run_dir, "model.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load config.py content
    CONFIG = load_config_from_path(config_path)
    print("Loaded config.")

    # Create model instance
    model = PINN(CONFIG).to(device)

    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    x = torch.linspace(0, 1, CONFIG["num_x_points"]).view(-1, 1).to(device)
    print("Loaded model.")

    model.train()
    return model, x, CONFIG, run_dir


if __name__ == "__main__":
    model, x, CONFIG, run_dir = load_latest_model()
    print("\nModel + config loaded successfully!")
