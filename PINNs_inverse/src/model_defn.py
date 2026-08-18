import torch
import torch.nn as nn
import numpy as np

def fourier_disk_features(x, modes=10):
    r = x[:,0]
    theta = x[:,1]
    features = [r.unsqueeze(1)]
    for k in range(1, modes+1):
        features.append(torch.sin(k*theta).unsqueeze(1))
        features.append(torch.cos(k*theta).unsqueeze(1))
        features.append((r*torch.sin(k*theta)).unsqueeze(1))
        features.append((r*torch.cos(k*theta)).unsqueeze(1))
    return torch.cat(features, dim=1)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.linear.in_features
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(
        self, num_modes=1, output_dim=1, num_hid_layers=3, hid_dim=32, omega_0=30, dropout_prob=0.05, device=None):
        super(SIREN, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        layers = []
        layers.append( SineLayer( 1 + 4*num_modes, hid_dim, is_first=True, omega_0=omega_0))
        for _ in range(num_hid_layers - 1):
            layers.append( SineLayer( hid_dim, hid_dim, omega_0=omega_0) )
            layers.append(nn.Dropout(p=dropout_prob))

        final_linear = nn.Linear(hid_dim, output_dim)

        with torch.no_grad():
            bound = np.sqrt(6 / hid_dim) / omega_0
            final_linear.weight.uniform_(-bound, bound)
            final_linear.bias.uniform_(-bound, bound)

        layers.append(final_linear)
        self.num_modes = num_modes
        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        features = fourier_disk_features(x, modes = self.num_modes)
        return self.model(features)

def load_model(CONFIG, output_dim= 1, device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SIREN(num_modes=CONFIG["num_modes"], output_dim=output_dim, num_hid_layers=CONFIG["num_lay"], hid_dim=CONFIG["hid_dim"], dropout_prob = CONFIG["drop_p"], device = device)
    return model


if __name__ == "__main__":
    CONFIG = {
        "num_modes": 1,
        "num_lay": 3,
        "hid_dim": 32,
        "drop_p": 0.1,
        "num_x_points": 20000,
        "num_y_points": 20000,
        }
    model = load_model(CONFIG, device = 'cpu')
    r = torch.sqrt(torch.rand(CONFIG['num_x_points'])).to('cpu')
    theta = 2*torch.pi*torch.rand(CONFIG['num_x_points']).to('cpu')
    x = torch.stack([r, theta], dim=1)
    TEST_out = model(x)
    print(TEST_out.shape)
