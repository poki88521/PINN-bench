import numpy as np
import deepxde as dde
import torch
from examples.ExampleObject import ExampleObject


class Helmholtz2D(ExampleObject):
    def __init__(self, dataset_dir, example_config):
        super().__init__()
        self.name = "Helmholtz2D"
        self.geom = dde.geometry.Rectangle([-1, -1], [1, 1])
        self.bcs = [dde.DirichletBC(self.geom, self.u_exact_numpy, lambda _, on_boundary: on_boundary)]
        self.exact_func = self.u_exact_numpy
        self.a1 = example_config.a1
        self.a2 = example_config.a2
        self.k = example_config.k

    def pde(self, x, u):
        # x: [N, 2]
        # u: [N, 1]
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f_pred = u_xx + u_yy + self.k ** 2 * u
        return f_pred - self.f_source(x)

    def f_source(self, x):
        u_xx = - (self.a1 * torch.pi) ** 2 * self.u_exact_torch(x)
        u_yy = - (self.a2 * torch.pi) ** 2 * self.u_exact_torch(x)
        return u_xx + u_yy + self.k ** 2 * self.u_exact_torch(x)

    def u_exact_numpy(self, x):
        return np.sin(self.a1 * np.pi * x[:, 0:1]) * np.sin(self.a2 * np.pi * x[:, 1:2])

    def u_exact_torch(self, x):
        return torch.sin(self.a1 * torch.pi * x[:, 0:1]) * torch.sin(self.a2 * torch.pi * x[:, 1:2])



