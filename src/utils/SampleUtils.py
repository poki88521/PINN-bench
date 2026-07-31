import deepxde as dde


class Sampler:
    def __init__(self, num_domain, num_boundary, num_test):
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_test = num_test

    def sample(self, example):
        data = dde.data.PDE(example.geom, example.pde, example.bc,
                            num_domain=self.num_domain, num_boundary=self.num_boundary,
                            solution=example.function, num_test=self.num_test)
        return data
