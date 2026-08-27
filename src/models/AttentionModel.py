from models import FNN
from deepxde.nn import initializers
from torch import nn
import torch

class AttentionNet(FNN):
    def __init__(self, dims, model_config):
        super(AttentionNet, self).__init__(dims, model_config)
        self.attentions = nn.ModuleList([nn.Linear(dims[0], dims[1]), nn.Linear(dims[0], dims[1])])
        # 把config变成初始化函数并调用（参考torch和deepxde的字典）
        initializers.get(model_config.init)(self.attentions[0].weight.data)
        nn.init.zeros_(self.attentions[0].bias.data)
        initializers.get(model_config.init)(self.attentions[1].weight.data)
        nn.init.zeros_(self.attentions[1].bias.data)


    def forward(self, x):
        #格式校验
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.get_default_dtype())
        #归一化
        if self._input_transform is not None:
            x = self._input_transform(x)
        encoders = [self.activation(self.attentions[0](x)), self.activation(self.attentions[1](x))]
        a = self.activation(self.layers[0](x))
        a = a * encoders[0] + (1 - a) * encoders[1]
        for i in range(1, len(self.layers) - 1):
            a = self.activation(self.layers[i](a))
            a = a * encoders[0] + (1 - a) * encoders[1]
        return self.layers[-1](a)


#校验网络结构的主函数已被移除
