import time
import os
import numpy as np
from deepxde import Model
import deepxde as dde
from models import FeedForwardNet



def path_init(config, example_dir):
    version_dir = os.path.join(example_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    model_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_model")
    history_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_history.csv")
    info_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_info.csv")
    return version_dir, model_path, history_path, info_path


def train(model : Model, training_config, model_path):
    model.compile(training_config.optimizer, lr=training_config.lr)
    time_start = time.time()
    loss_history, train_state = model.train(iterations=training_config.iterations,
                                            batch_size=training_config.batch_size,
                                            model_save_path=model_path,
                                            display_every=training_config.display_every)
    time_elapsed = time.time() - time_start
    print(f"elapsed time: {time_elapsed:.2f}s")
    return loss_history, train_state, model


def create_dataset(example, data_config):
    data = example.get_data(data_config.num_domain, data_config.num_boundary, data_config.num_test)
    return data


def create_model(data, dims_config):
    dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]
    return dde.Model(data, FeedForwardNet(dims))


def test(example, model, data_config):
    x_test = example.geom.uniform_points(n=data_config.num_test, boundary=True)
    u_true = example.u_exact_numpy(x_test)
    u_pred = model.predict(x_test)
    error = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {error:.2e}")
    pass