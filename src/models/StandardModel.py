import torch
import deepxde as dde
from torch import nn
import torchinfo

class FNN(dde.nn.pytorch.NN):
    def __init__(self, dims):
        super(FNN, self).__init__()
        self.dims = dims
        self.layers = self.create_layers(dims)
        self.activation = nn.Tanh()

    def create_layers(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            nn.init.xavier_normal_(layers[i].weight.data, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layers[i].bias.data)
        return nn.ModuleList(layers)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        for i in range(len(self.layers) - 2):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)


if __name__ == '__main__':
    dims = [2] + 5 * [70] + [1]
    model = FNN(dims)
    torchinfo.summary(model, input_size=(1, 2))

