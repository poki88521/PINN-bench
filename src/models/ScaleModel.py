import torch
from deepxde.nn import initializers
from torch import nn


#Scale-PINN 网络：首层 sin(2*pi*·) 注入高频，隐藏层 silu，输出层线性（无激活）
#纯 torch 实现，不接入 deepxde 的 Model 体系，供 ScaleTrainer 的训练循环使用
class ScaleNet(nn.Module):
    def __init__(self, dims, model_config):
        super().__init__()
        self.dims = dims
        self.model_config = model_config
        self.layers = self.create_layers()

    #按 dims 建全连接链（含输出层），权重按 config.init 初始化、bias 置零
    def create_layers(self):
        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            initializers.get(self.model_config.init)(layers[i].weight.data)
            nn.init.zeros_(layers[i].bias.data)
        return nn.ModuleList(layers)

    #网络主体：输入到 u 的映射（输出层不带激活，与官方 last_layer 一致）
    def compute_u(self, x):
        x = torch.sin(2 * torch.pi * self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = torch.nn.functional.silu(layer(x))
        return self.layers[-1](x)

    #前向：返回 u、u_xx、u_t
    def forward(self, input):
        #输入列序约定：[x, t]；点池张量默认不需梯度，此处开启以支持求导
        if not input.requires_grad:
            input = input.detach().requires_grad_(True)
        u = self.compute_u(input)
        #一阶导，du 的两列依次为 du/dx、du/dt（与列序[x, t]对应）
        #必须 create_graph=True：否则 du 无 grad_fn，无法继续求二阶导
        du = torch.autograd.grad(u, input, torch.ones_like(u),
                                 create_graph=True)[0]
        u_t = du[:, 1:2]
        #二阶导：u_xx = d²u/dx²
        u_xx = torch.autograd.grad(du[:, 0:1], input, torch.ones_like(u),
                                   create_graph=True)[0][:, 0:1]
        return {"u": u, "u_xx": u_xx, "u_t": u_t}

    #推理用：只算 u，不构建计算图
    def predict_u(self, inputs):
        with torch.no_grad():
            return self.compute_u(inputs)
