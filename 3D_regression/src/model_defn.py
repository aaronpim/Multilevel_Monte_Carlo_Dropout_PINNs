import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_dim=2, output_dims=[60,60,60], num_hid_layers=3, hid_dim=64, activation="nn.ReLU()", dropout_prob = 0.05, device=None):
        super(RegressionModel, self).__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            activation = eval(activation, {"nn": nn})
        except:
            activation = activation
        self.output_dims = output_dims
        final_output_dim = 1
        for d in output_dims:
            final_output_dim *= d
        layers = [nn.Linear(input_dim, hid_dim), activation, nn.Dropout(p=dropout_prob)]
        for _ in range(num_hid_layers):
            layers.append( nn.Linear(hid_dim, hid_dim) )
            layers.append( activation )
            layers.append(nn.Dropout(p=dropout_prob))

        layers.append( nn.Linear(hid_dim, final_output_dim) )
        self.model = nn.Sequential(*layers)
        self.to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.model(x)
        return y.view(-1, *self.output_dims)

def load_model(CONFIG, input_dim = 2, output_dims=[60,60,60], device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return RegressionModel(input_dim=input_dim,
                           output_dims=output_dims,
                           num_hid_layers=CONFIG["num_hid_layers"],
                           hid_dim=CONFIG["hid_dim"],
                           activation=CONFIG["activation"],
                           dropout_prob = CONFIG["dropout_prob"],
                           device = device)


if __name__ == "__main__":
    x = torch.rand(55,2).to("cpu")
    model = RegressionModel(device = "cpu")
    y = model(x)-model(x)
    print(y.shape)
