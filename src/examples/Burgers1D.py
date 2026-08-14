import deepxde as dde
import math
import numpy as np
import os
import scipy
from examples.ExampleObject import ExampleObject


class Burgers1D(ExampleObject):
    def __init__(self, dataset_dir, example_config):
        super().__init__()
        self.name = 'Burgers1D'
        self.nu = 0.01 / math.pi
        self.geom = dde.geometry.Interval(-1, 1)
        self.time_domain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.time_domain)
        self.bcs = [dde.DirichletBC(self.geomtime, lambda x: 0,
                                     lambda x, on_boundary: on_boundary and np.isclose(x[0], -1)),
                    dde.DirichletBC(self.geomtime, lambda x: 0,
                                    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1))]
        self.ics = [dde.IC(self.geomtime, self.ic_u, lambda _, on_initial: on_initial)]
        self.exact_func = None
        self.dataset_path = os.path.join(dataset_dir, 'Burgers.npz')

    def pde(self, x, u):
        u_t = dde.grad.jacobian(u, x, j=1)
        u_x = dde.grad.jacobian(u, x, j=0)
        u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        return u_t + u * u_x - self.nu * u_xx

    def ic_u(self, x):
        return - np.sin(np.pi * x[:, 0:1])

    def load_data(self, x_test):
        npz = np.load(self.dataset_path)
        t = np.ravel(npz['t'])
        x = np.ravel(npz['x'])
        u = npz['usol']
        interp = scipy.interpolate.RegularGridInterpolator(
            (x, t), u, bounds_error=False, fill_value=None)
        u_true = interp(x_test)
        return u_true.reshape(-1, 1)
