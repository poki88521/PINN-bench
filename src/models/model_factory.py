import deepxde as dde
from models import *


#模型调度器
def create_model(config, data):
    dims_config = config.dims
    model_name = config.model.model_name
    dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]
    if model_name == "FNN":
        return dde.Model(data, FNN(dims, config.model))
    elif model_name == "AttentionNet":
        return dde.Model(data, AttentionNet(dims, config.model))
    else:
        raise ValueError("Unknown model")