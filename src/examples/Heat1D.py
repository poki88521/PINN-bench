import deepxde as dde
import numpy as np
import torch
from examples.ExampleObject import ExampleObject

#算例具体信息见examples.md
class Heat1D(ExampleObject):
    def __init__(self, dataset_dir, example_config):
        super().__init__()
        self.name = "Heat1D"
        self.geom = dde.geometry.Interval(0, 1)
        self.time_domain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.time_domain)
        self.bcs = [dde.DirichletBC(self.geomtime, lambda x: 0, lambda x, on_boundary: on_boundary)]
        self.ics = [dde.IC(self.geomtime, self.ic_func, lambda x, on_initial: on_initial)]
        self.exact_func = self.u_exact_numpy
        self.alpha = example_config.alpha
        pass

    def pde(self, x, u):
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        return u_t - self.alpha * u_xx

    def ic_func(self, x):
        return np.sin(np.pi * x[:, 0:1])

    def u_exact_numpy(self, x):
        # x: [N, 2] (space = [:, 0], time = [:, 1])
        return np.exp(- (np.pi ** 2) * self.alpha * x[:, 1:2]) * np.sin(np.pi * x[:, 0:1])

    def u_exact_torch(self, x):
        return torch.exp(- torch.pi ** 2 * self.alpha * x[:, 1:2]) * torch.sin(torch.pi * x[:, 0:1])