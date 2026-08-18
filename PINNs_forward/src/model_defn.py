import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, activation=nn.SiLU(), sigma=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        B = torch.randn(input_dim, embedding_dim).to(device) * self.sigma
        self.register_buffer("B", B)
        self.u_trans = nn.Linear(2*embedding_dim, output_dim)
        self.v_trans = nn.Linear(2*embedding_dim, output_dim)
        self.actfunc = activation

    def forward(self, x):
        sin_part = torch.sin(torch.matmul(x, self.B))
        cos_part = torch.cos(torch.matmul(x, self.B))
        phi_vec  = torch.cat([cos_part, sin_part], dim=-1)
        U = self.actfunc(self.u_trans(phi_vec))
        V = self.actfunc(self.v_trans(phi_vec))
        return U, V

class PirateNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=nn.SiLU(), p_drop = 0.05):
        super(PirateNetBlock, self).__init__()
        self.actfunc = activation
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, input_dim)
        self.alpha = nn.parameter.Parameter(torch.zeros(1)-3)
        self.drop_layer = nn.Dropout(p = p_drop)

    def forward(self, x, U, V):
        f = self.drop_layer(self.actfunc(self.W1(x)))
        z1 = f * U + (1 - f) * V
        g = self.drop_layer(self.actfunc(self.W2(z1)))
        z2 = g * U + (1 - g) * V
        h = self.actfunc(self.W3(z2))

        out = torch.sigmoid(self.alpha) * h + (1 - torch.sigmoid(self.alpha)) * x
        return out

class PirateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks = 3, output_dim = None, embedding_dim = None, p_drop = 0.05, activation=nn.SiLU(), sigma=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PirateNet, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        if embedding_dim is None:
            embedding_dim = hidden_dim
        self.embedding = FourierEmbedding(input_dim, embedding_dim, hidden_dim, activation, sigma, device)
        self.blocks = nn.ModuleList([PirateNetBlock(input_dim, hidden_dim, activation, p_drop) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        U, V = self.embedding(x)
        for block in self.blocks:
            x = block(x, U, V)
        output = self.output_layer(x)
        return output

def load_PirateNet(CONFIG, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = PirateNet(
        input_dim  = CONFIG["input_dim"],
        hidden_dim = CONFIG["hidden_dim"],
        num_blocks = CONFIG["num_blocks"],
        output_dim = CONFIG["output_dim"],
        p_drop     = CONFIG["p_drop"],
        activation = eval(CONFIG["activation"], {"nn": nn}),
        sigma = CONFIG["sigma"]
        )
    return model.to(device)

if __name__ == "__main__":
    model = PirateNet(input_dim = 1, hidden_dim = 32, p_drop = 0.05, device = "cpu")
    print(model.state_dict().keys())
    model.train()
    x = torch.tensor([0.0])
    s1 = model(x)
    s2 = model(x)
    print(s1 - s2)
