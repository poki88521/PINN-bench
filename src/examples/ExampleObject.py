import deepxde as dde

class ExampleObject:
    def __init__(self):
        self.name = "ExampleObject"
        self.geom = None
        self.time_domain = None
        self.geomtime = None
        self.bcs = []
        self.ics = []
        self.exact_func = None

    def pde(self, x, u):
        raise NotImplementedError()

    def get_data(self, num_domain, num_boundary, num_test, num_initial=None):
        if not self.ics:
            return dde.data.PDE(self.geom, self.pde, self.bcs,
                                num_domain=num_domain, num_boundary=num_boundary,
                                solution=self.exact_func, num_test=num_test)
        else:
            return dde.data.TimePDE(self.geomtime, self.pde, self.bcs + self.ics,
                                    num_domain=num_domain, num_boundary=num_boundary,
                                    num_initial=num_initial,
                                    solution=self.exact_func, num_test=num_test)