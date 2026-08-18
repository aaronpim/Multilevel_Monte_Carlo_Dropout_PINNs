import torch
import time
import matplotlib.pyplot as plt
from src.model_defn import load_PirateNet

def gen_x_and_eps(CONFIG, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    x = torch.linspace(0,1, CONFIG["x_num"]).to(device)
    eps = torch.linspace(CONFIG["eps_min"],CONFIG["eps_max"], CONFIG["eps_num"]).to(device)
    X, E = torch.meshgrid(x,eps, indexing = 'ij')
    X = X.requires_grad_(True)
    input_pairs = torch.stack([X, E.detach() ], dim = -1)
    return input_pairs, X, E, x, eps

def gen_boundary_x_and_eps(CONFIG, device = 'cpu'):
    x = torch.tensor([0,1]).to(device)
    x_expanded = x.expand(CONFIG["eps_num"], 2).T
    log_eps_vec = torch.sort(torch.rand(CONFIG["eps_num"]))[0].to(device)*(CONFIG["eps_max"] - CONFIG["eps_min"]) + CONFIG["eps_min"]
    log_epsilon_expanded = log_eps_vec.expand(2, CONFIG["eps_num"])
    input_pairs = torch.stack([x_expanded, log_epsilon_expanded], dim=-1)
    return input_pairs, x_expanded, log_epsilon_expanded, x, log_eps_vec

def exact_soln(x, log_eps):
    normalising = torch.exp(torch.exp(-log_eps/2)) + 1
    f = torch.exp(x / torch.exp(log_eps/2) ) + torch.exp((1 - x) / torch.exp(log_eps/2) )
    return 1 - f / normalising

def test_exact_soln():
    CONFIG = {
        "x_num": 301,
        "eps_num": 201,
        "eps_min": -8,
        "eps_max": 4,
        }
    _, x_expanded, log_epsilon_expanded, _, _ = gen_x_and_eps(CONFIG, device = "cpu")
    F = exact_soln(x_expanded, log_epsilon_expanded)
    plt.contourf(x_expanded.detach().numpy(), log_epsilon_expanded.detach().numpy(), F.detach().numpy(), levels=50, cmap='viridis')
    plt.colorbar()
    plt.show()

def estimate_error(model, CONFIG, device='cuda' if torch.cuda.is_available() else 'cpu'):
    with torch.no_grad():
        input_pairs, x_expanded, log_epsilon_expanded, x, log_eps_vec = gen_x_and_eps(CONFIG, device = device)
        U = exact_soln(x_expanded, log_epsilon_expanded)
        N = model(input_pairs).squeeze()
        diff = (U-N).pow(2)
        err = torch.trapezoid( torch.trapezoid(diff, x, dim = 0), log_eps_vec)
    return torch.sqrt(err)

def pinns_loss(model, CONFIG, device='cuda' if torch.cuda.is_available() else 'cpu'):
    input_pairs, x_expanded, log_epsilon_expanded, x, log_eps_vec = gen_x_and_eps(CONFIG, device)
    model = model.to(device)
    u = model(input_pairs).squeeze()
    u_x  = torch.autograd.grad(u,   x_expanded, grad_outputs = torch.ones_like(u),   create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_expanded, grad_outputs = torch.ones_like(u_x), create_graph=True)[0]
    res  = (u - torch.exp(log_epsilon_expanded)*u_xx - 1.0).pow(2)
    integrand = torch.trapezoid(res, x, dim = 0)
    return torch.trapezoid(integrand, log_eps_vec)

def bcs_loss(model, CONFIG, device='cuda' if torch.cuda.is_available() else 'cpu'):
    input_pairs, _, _, _, _ = gen_boundary_x_and_eps(CONFIG, device)
    model = model.to(device)
    u = model(input_pairs).squeeze()
    integrand = torch.sum(u.pow(2), dim = 0)
    return integrand.mean()

def test_PINNs():
    CONFIG = {
        "x_num": 301,
        "eps_num": 201,
        "eps_min": -8,
        "eps_max": 4,
        "input_dim": 2,
        "hidden_dim": 32,
        "num_blocks": 4,
        "output_dim": 1,
        "p_drop": 0.05,
        "activation": "nn.ReLU()",
        "sigma": 1.0
        }
    model = load_PirateNet(CONFIG, device = "cpu")
    print(pinns_loss(model, CONFIG, device = "cpu"))
    print(bcs_loss(model, CONFIG, device = "cpu"))
    print(estimate_error(model, CONFIG, device = "cpu"))

if __name__ == "__main__":
    test_exact_soln()
    test_PINNs()
