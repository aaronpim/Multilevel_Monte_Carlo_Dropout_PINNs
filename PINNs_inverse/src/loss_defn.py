import time
import torch
import matplotlib.pyplot as plt
from src.model_defn import load_model

def get_device(model):
    return next(model.parameters()).device

def Greens(x, y, clamp_val = 1e-12):
    ###
    # inputs:
    #   x - [Nx2] torch tensor representing polar coordinates on unit disk
    #   y - [Mx2] torch tensor representing polar coordinates on unit disk
    #
    # output:
    #   G - [NxM] torch tensor representing Green's function
    ###
    rx = x[:, 0].unsqueeze(1)
    tx = x[:, 1].unsqueeze(1)
    ry = y[:, 0].unsqueeze(0)
    ty = y[:, 1].unsqueeze(0)
    dtheta = tx - ty
    den = rx**2 + ry**2 - 2*rx*ry*torch.cos(dtheta)
    num = 1 + (rx*ry)**2 - 2*rx*ry*torch.cos(dtheta)
    G = (1/(4*torch.pi)) * torch.log(num.clamp(clamp_val) / den.clamp(clamp_val))
    return G

def random_disk_points(N, device = None):
    ###
    # Generate N uniform random points in the unit disk.
    # Returns polar coordinates [r,theta].
    ###
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    r = torch.sqrt(torch.rand(N)).to(device)
    theta = 2*torch.pi*torch.rand(N).to(device)
    return torch.stack([r, theta], dim=1)

def test_Greens():
    torch.manual_seed(0)
    x = random_disk_points(20000, device = 'cpu')
    y = random_disk_points(20000, device = 'cpu')
    G = Greens(x, y)
    u = torch.pi * torch.mean(G, dim=1)
    xc = x[:,0] * torch.cos(x[:,1])
    yc = x[:,0] * torch.sin(x[:,1])
    u_exact = (1 - xc**2 - yc**2) / 4
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sc0 = axes[0].scatter(xc.numpy(),yc.numpy(),c=u.numpy())
    axes[0].set_title("Estimated Solution")
    axes[0].axis("equal")
    fig.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(xc.numpy(),yc.numpy(),c=(u - u_exact).abs().numpy())
    axes[1].set_title("Absolute Error")
    axes[1].axis("equal")
    fig.colorbar(sc1, ax=axes[1])
    sc2 = axes[2].scatter(xc.numpy(),yc.numpy(),c=((u - u_exact).abs() / (u_exact.abs())).numpy())
    axes[2].set_title("Relative Error")
    axes[2].axis("equal")
    fig.colorbar(sc2, ax=axes[2])
    plt.tight_layout()
    plt.show()

def loss(model, x, data, CONFIG):
    device = get_device(model)
    y = random_disk_points(CONFIG['num_y_points'], device = device)
    f = model(y).T
    G = Greens(x, y, clamp_val = CONFIG["clamp"])
    u = torch.pi * torch.mean(G*f, dim=1)
    res = (u-data)**2
    if CONFIG["smoothing_coef"] > 0.0:
        y = y.requires_grad_()
        f = model(y)
        grad_f = torch.autograd.grad(outputs=f, inputs=y, grad_outputs=torch.ones_like(f), create_graph=True)[0]
        r = y[:,0].clamp(CONFIG["clamp"])
        Q = grad_f[:,0]**2 + (grad_f[:,1]/r)**2
        return torch.pi * (torch.mean(res) + CONFIG["smoothing_coef"] * torch.mean(Q))
    else:
        return torch.pi * torch.mean(res)

def test_loss():
    CONFIG = {
        "num_modes": 1,
        "num_lay": 3,
        "hid_dim": 128,
        "drop_p": 0.1,
        "smoothing_coef":0.1,
        "num_x_points": 2000,
        "num_y_points": 2000,
        "num_drop_evals": 10,
        "clamp": 1e-12,
        }
    torch.manual_seed(0)
    model = load_model(CONFIG, device = 'cpu')
    x = random_disk_points(CONFIG["num_x_points"], device = get_device(model) )
    data = (1 - x[:,0]**2) / 4
    start = time.time()
    l = 0
    for _ in range(CONFIG["num_drop_evals"]):
        l += loss(model, x, data, CONFIG)/CONFIG["num_drop_evals"]
    l.backward()
    print(time.time()-start)

if __name__ == "__main__":
    #test_Greens()
    test_loss()

