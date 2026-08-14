import deepxde as dde
import numpy as np
import torch
from examples.ExampleObject import ExampleObject


class Wave1D(ExampleObject):
    def __init__(self, dataset_dir, example_config):
        super().__init__()
        self.name = "Wave1D"
        self.geom = dde.geometry.Interval(0, 1)
        self.time_domain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.time_domain)
        self.bcs = [dde.DirichletBC(self.geomtime, lambda x: 0,
                                     lambda x, on_boundary: on_boundary and np.isclose(x[0], 0)),
                    dde.DirichletBC(self.geomtime, lambda x: 0,
                                    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1))]
        self.ics = [dde.IC(self.geomtime, self.ic_u, lambda _, on_initial: on_initial),
                    dde.OperatorBC(self.geomtime, self.ic_u_t, lambda x, _: np.isclose(x[1], 0))]
        self.exact_func = self.u_exact_numpy
        self.c = example_config.c

    def pde(self, x, u):
        u_tt = dde.grad.hessian(u, x, component=0, i=1, j=1)
        u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        return u_tt - self.c ** 2 * u_xx

    def ic_u(self, x):
        return np.sin(np.pi * x[:, 0:1])

    def ic_u_t(self, x, u, X):
        return dde.grad.jacobian(u, x, i=0, j=1)

    def u_exact_numpy(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.cos(self.c * np.pi * x[:, 1:2])

    def u_exact_torch(self, x):
        return torch.sin(torch.pi * x[:, 0:1]) * torch.cos(self.c * torch.pi * x[:, 1:2])

