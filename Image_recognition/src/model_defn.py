import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

class MNIST_CNN(nn.Module):
    def __init__(self, drop_p = 0.1, num_conv_layers = 3, width = 32, MLP_width = 128, kernal_size = 3, padding = 1,  device=None):
        super(MNIST_CNN, self).__init__()
        self.input_layer = nn.Conv2d(1, width, kernel_size=kernal_size, padding=padding)
        self.conv_layers = nn.ModuleList([ nn.Conv2d(width, width, kernel_size=kernal_size, padding=padding) for _ in range(num_conv_layers) ])
        self.pool        = nn.MaxPool2d(2, 2)
        self.drop        = nn.Dropout2d(p=drop_p)
        self.out_layer_1 = nn.LazyLinear(MLP_width)
        self.out_layer_2 = nn.Linear(MLP_width, 10)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if isinstance(m.weight, UninitializedParameter):
                    continue
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.conv_layers:
            x = F.relu(layer(x))
            x = self.drop(x)
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.out_layer_1(x))
        x = self.out_layer_2(x)
        return x

def load_model(CONFIG, device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return MNIST_CNN(drop_p          = CONFIG["drop_p"],
                     num_conv_layers = CONFIG["num_conv_layers"],
                     width           = CONFIG["width"],
                     MLP_width       = CONFIG["MLP_width"],
                     kernal_size     = CONFIG["kernal_size"],
                     padding         = CONFIG["padding"],
                     device          = device)

if __name__ == "__main__":
    x = torch.rand(3, 1, 28,28).to("cpu")
    model = MNIST_CNN(device = "cpu")
    y = model(x)
    print(y.shape)
