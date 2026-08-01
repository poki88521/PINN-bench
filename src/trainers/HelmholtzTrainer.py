import time
import deepxde as dde
import numpy as np
import os
from examples import Helmholtz2D
from models import FeedForwardNet
from utils import Sampler, load_yaml


def train_init(config_dict):
    example_config = config_dict.example
    sampler_config = config_dict.sampler
    dims_config = config_dict.dims
    training_config = config_dict.training
    example = Helmholtz2D(example_config.a1, example_config.a2, example_config.k)
    sampler = Sampler(sampler_config.num_domain, sampler_config.num_boundary, sampler_config.num_test)
    dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]
    return example, dims, sampler, training_config

def path_init(config, example_dir):
    version_dir = os.path.join(example_dir, config.version)
    model_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_model.pth")
    return version_dir, model_path

def train(example, dims, sampler, training_config, model_path):
    data = sampler.sample(example)
    net = FeedForwardNet(dims)
    model = dde.Model(data, net)
    model.compile("adam", lr=training_config.lr)

    time_start = time.time()
    loss_history, train_state = model.train(iterations=training_config.iterations,
                                            batch_size=training_config.batch_size,
                                            model_save_path=model_path)
    time_elapsed = time.time() - time_start
    u_true = example.u_exact(data.test_x)
    u_pred = model.predict(data.test_x)
    error = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {error:.2e}, Time: {time_elapsed:.2f}s")
    return loss_history, train_state

def evaluate(loss_history, train_state):
    pass


def launch(path, example_dir):
    config = load_yaml(path)
    example, dims, sampler, training_config = train_init(config)
    version_dir, model_path = path_init(config, example_dir)
    loss_history, train_state = train(example, dims, sampler, training_config, model_path)
