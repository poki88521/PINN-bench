from models import FNN
from torch import nn
import torchinfo
import torch

class AttentionNet(FNN):
    def __init__(self, dims):
        super(AttentionNet, self).__init__(dims)
        self.attentions = nn.ModuleList([nn.Linear(dims[0], dims[1]), nn.Linear(dims[0], dims[1])])
        nn.init.xavier_normal_(self.attentions[0].weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.attentions[0].bias.data)
        nn.init.xavier_normal_(self.attentions[1].weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.attentions[1].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.get_default_dtype())
        encoders = [self.activation(self.attentions[0](x)), self.activation(self.attentions[1](x))]
        a = self.activation(self.layers[0](x))
        a = a * encoders[0] + (1 - a) * encoders[1]
        for i in range(1, len(self.layers) - 1):
            a = self.activation(self.layers[i](a))
            a = a * encoders[0] + (1 - a) * encoders[1]
        return self.layers[-1](a)


if __name__ == '__main__':
    dims = [2] + 5 * [70] + [1]
    model_attention = AttentionNet(dims)
    torchinfo.summary(model_attention, input_size=(1, 2))
