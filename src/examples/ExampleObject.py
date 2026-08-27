import deepxde as dde

#所有算例类的父类
class ExampleObject:
    def __init__(self):
        self.name = "ExampleObject"
        #几何区域、时间区域、时空区域变量（都用dde的）
        self.geom = None
        self.time_domain = None
        self.geomtime = None
        #边界条件和初始条件列表
        self.bcs = []
        self.ics = []
        #精确解（如果有）
        self.exact_func = None

    #返回一个方程的loss给训练用
    def pde(self, x, u):
        raise NotImplementedError()

    #获取数据（要分稳态问题和瞬态问题）
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