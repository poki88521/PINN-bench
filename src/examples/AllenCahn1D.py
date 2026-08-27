import deepxde as dde
import numpy as np
import os
import scipy
from examples.ExampleObject import ExampleObject


#算例具体信息见examples.md
class AllenCahn1D(ExampleObject):
    def __init__(self, dataset_dir, example_config):
        super().__init__()
        self.name = "AllenCahn1D"
        self.d = example_config.d
        self.geom = dde.geometry.Interval(-1, 1)
        self.time_domain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.time_domain)
        self.bcs = [dde.DirichletBC(self.geomtime, lambda x: -1,
                                     lambda x, on_boundary: on_boundary and np.isclose(x[0], -1)),
                    dde.DirichletBC(self.geomtime, lambda x: -1,
                                    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1))]
        self.ics = [dde.IC(self.geomtime, self.ic_u, lambda _, on_initial: on_initial)]
        self.exact_func = None
        self.dataset_path = os.path.join(dataset_dir, "Allen_Cahn.mat")

    def pde(self, x, u):
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        return u_t - self.d * u_xx - 5 * (u - u**3)

    def ic_u(self, x):
        return x[:, 0:1] ** 2 * np.cos(np.pi * x[:, 0:1])

    def load_data(self, x_test):
        dataset = scipy.io.loadmat(self.dataset_path)
        t, x, u = dataset['t'], dataset['x'], dataset['u']
        t = np.ravel(t)
        x = np.ravel(x)
        idx_t = np.argsort(t)
        idx_x = np.argsort(x)
        t = t[idx_t]
        x = x[idx_x]
        u = u[np.ix_(idx_t, idx_x)]
        interp = scipy.interpolate.RegularGridInterpolator(
            (x, t), u.T, bounds_error=False, fill_value=None
        )
        u_true = interp(x_test)
        return u_true.reshape(-1, 1)

