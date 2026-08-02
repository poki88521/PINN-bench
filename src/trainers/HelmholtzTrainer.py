import time
import deepxde as dde
import numpy as np
import os
import csv
from examples import Helmholtz2D
from models import FeedForwardNet
from utils import Sampler, load_yaml


def train_init(config_dict):
    example_config = config_dict.example
    data_config = config_dict.data
    dims_config = config_dict.dims
    training_config = config_dict.training
    example = Helmholtz2D(example_config.a1, example_config.a2, example_config.k)
    dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]
    return data_config, training_config, example, dims

def path_init(config, example_dir):
    version_dir = os.path.join(example_dir, config.version)
    model_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_model.pth")
    history_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_history.csv")
    return version_dir, model_path, history_path

def create_dataset(example, data_config):
    data = example.get_data(data_config.num_domain, data_config.num_boundary, data_config.num_test)
    return data

def create_model(data, dims):
    return dde.Model(data, FeedForwardNet(dims))

def train(model, training_config, model_path):
    model.compile("adam", lr=training_config.lr)
    time_start = time.time()
    loss_history, train_state = model.train(iterations=training_config.iterations,
                                            batch_size=training_config.batch_size,
                                            model_save_path=model_path)
    time_elapsed = time.time() - time_start
    print(f"Time: {time_elapsed:.2f}s")
    return loss_history, train_state, model

def test(example, model, data_config):
    x_test = example.geom.uniform_points(num=data_config.num_test, boundary=True)
    u_true = example.u_exact_numpy(x_test)
    u_pred = model.predict(x_test)
    error = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {error:.2e}")
    pass

def evaluate(loss_history, train_state, history_path, training_config):
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'train_loss', 'test_loss'])
        i = 0
        while i <= training_config.iterations:
            if i % training_config.print_every == 0:
                writer.writerow([i, loss_history.loss_train[i], loss_history.loss_test[i]])
            i += 1
    pass


def launch(path, example_dir):
    config = load_yaml(path)
    data_config, training_config, example, dims = train_init(config)
    version_dir, model_path, history_path = path_init(config, example_dir)
    data = create_dataset(example, data_config)
    model = create_model(data, dims)
    loss_history, train_state, model = train(model, training_config, model_path)
    test(example, model, data_config)
    evaluate(loss_history, train_state, history_path, training_config)
