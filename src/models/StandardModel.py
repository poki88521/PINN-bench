import torch
import deepxde as dde
from deepxde.nn import activations, initializers
from torch import nn
import torchinfo

class FNN(dde.nn.pytorch.NN):
    def __init__(self, dims, model_config):
        super(FNN, self).__init__()
        self.dims = dims
        self.layers = self.create_layers(dims, model_config)
        self.activation = activations.get(model_config.activation)

    def create_layers(self, dims, model_config):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            #把config变成初始化函数并调用（参考torch和deepxde的字典）
            initializers.get(model_config.init)(layers[i].weight.data)
            nn.init.zeros_(layers[i].bias.data)
        return nn.ModuleList(layers)

    def forward(self, x):
        #格式校验
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.get_default_dtype())
        #归一化
        if self._input_transform is not None:
            x = self._input_transform(x)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)


if __name__ == '__main__':
    #dims = [2] + 5 * [70] + [1]
    #model = FNN(dims, config)
    #torchinfo.summary(model, input_size=(1, 2))
    ...
