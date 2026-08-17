import deepxde as dde
from models import *
from models import AttentionNet


def create_model(model_name, dims_config, data):
    dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]
    if model_name == "FNN":
        return dde.Model(data, FNN(dims))
    elif model_name == "AttentionNet":
        return dde.Model(data, AttentionNet(dims))
    else:
        raise ValueError("Unknown model")