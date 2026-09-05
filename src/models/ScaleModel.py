import torch
from torch import nn
import deepxde as dde

from examples import AllenCahn1D


class PINN(nn.Module):
    def __init__(self, dims, model_config):
        super().__init__()
        self.dims = dims
        self.model_config = model_config
        self.layers = self.create_layers()
        self.last_model = None

    def create_layers(self):
        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            nn.init.zeros_(layers[i].bias.data)
        return nn.ModuleList(layers)

    def forward(self, x):
        x = torch.sin(2 * torch.pi * self.layers[0](x))
        for layer in self.layers[1:]:
            x = nn.SiLU(layer(x))
        u = self.last_model(x)
        u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        pde = AllenCahn1D(None, None).pde(x, u)
        return {"u": u, "u_xx": u_xx,}

    def pde(self, x, u):
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        return u_t - self.d * u_xx - 5 * (u - u**3)



class DNN(nn.Module):
    def __init__(self, dims, model_config):
        super().__init__()
