import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def exact_u(x):
    x1 = x[:,0]*torch.cos(x[:,1])
    x2 = x[:,0]*torch.sin(x[:,1])
    normalisation_c  = math.sqrt(24/math.pi)
    return x1*x2*(1 - x[:,0]**2)*normalisation_c/12

def exact_f(x):
    x1 = x[:,0]*torch.cos(x[:,1])
    x2 = x[:,0]*torch.sin(x[:,1])
    normalisation_c  = math.sqrt(24/math.pi)
    return x1*x2*normalisation_c

def sunflower_disk_points(n, device="cpu"):
    k = torch.arange(n, dtype=torch.float32, device=device)
    r = torch.sqrt(k / n)
    theta = k * math.pi * (3 - math.sqrt(5))
    return torch.stack((r, theta), dim=1)

def plot_values(inputs, u, f):
    r = inputs[:, 0].numpy()
    theta = inputs[:, 1].numpy()

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    tpc = axs[0].tripcolor(x, y, u.numpy(), shading="gouraud", cmap="viridis")
    axs[0].set_aspect("equal")
    axs[0].set_title("Exact $u$")
    fig.colorbar(tpc, ax=axs[0])

    tpc = axs[1].tripcolor(x, y, f.numpy(), shading="gouraud", cmap="viridis")
    axs[1].set_aspect("equal")
    axs[1].set_title("Exact $f$")
    fig.colorbar(tpc, ax=axs[1])

    plt.tight_layout()
    plt.savefig('exact_plots.png')
    plt.savefig('exact_plots.pdf')
    plt.close()

if __name__ == "__main__":
    inputs = sunflower_disk_points(10000)
    u = exact_u(inputs)
    f = exact_f(inputs)
    plot_values(inputs, u, f)
