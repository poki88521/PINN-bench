import numpy as np
import deepxde as dde


class Helmholtz2D:
    def __init__(self, a1=2, a2=4, k=1.0):
        self.name = "Helmholtz2D"
        self.geom = dde.geometry.Rectangle([-1, -1], [1, 1])
        self.bc = dde.DirichletBC(self.geom, Helmholtz2D.u_exact, lambda _, on_boundary: on_boundary)
        self.function = Helmholtz2D.u_exact
        self.a1 = a1
        self.a2 = a2
        self.k = k

    def pde(self, x, u):
        #x: [N, 2]
        #u: [N, 1]
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_yy = dde.grad.hessian(u, x, i=1, j=1)
        f_pred = u_xx + u_yy + self.k * u
        return f_pred - Helmholtz2D.f_source(self, x)

    @staticmethod
    def u_exact(x):
        a1 = 2
        a2 = 4
        return np.sin(a1 * np.pi * x[:, 0:1]) * np.sin(a2 * np.pi * x[:, 1:2])

    def u_func(self, x):
        return np.sin(self.a1 * np.pi * x[:, 0:1]) * np.sin(self.a2 * np.pi * x[:, 1:2])

    def f_source(self, x):
        u_xx = - (self.a1 * np.pi) ** 2 * Helmholtz2D.u_func(self, x)
        u_yy = - (self.a2 * np.pi) ** 2 * Helmholtz2D.u_func(self, x)
        return u_xx + u_yy + self.k ** 2 * Helmholtz2D.u_func(self, x)
