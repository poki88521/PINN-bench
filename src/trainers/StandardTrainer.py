import time
import os
import csv
from deepxde import Model
import deepxde as dde
from deepxde.model import TrainState
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
    if data_config.num_initial == 0:
        data = example.get_data(data_config.num_domain, data_config.num_boundary, data_config.num_test)
    else:
        data = example.get_data(data_config.num_domain, data_config.num_boundary,
                                data_config.num_initial, data_config.num_test)
    return data



def create_model(data, dims_config):
    dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]
    return dde.Model(data, FeedForwardNet(dims))


def save(loss_history, train_state : TrainState, history_path, training_config, info_path):
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'train_loss', 'test_loss'])
        #print(f"total num of loss:{len(loss_history.loss_train)}")
        for i, (tr, te) in enumerate(zip(loss_history.loss_train, loss_history.loss_test)):
            writer.writerow([i * training_config.display_every, tr, te])

    with open(info_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_step', "best_loss_train", "best_loss_test"])
        writer.writerow([train_state.best_step, train_state.best_loss_train, train_state.best_loss_test])
    pass


def launch(config, example, example_dir, test_func):
    version_dir, model_path, history_path, info_path = path_init(config, example_dir)
    data = create_dataset(example, config.data)
    model = create_model(data, config.dims)
    loss_history, train_state, model = train(model, config.training, model_path)
    test_func(example, model, config.data)
    save(loss_history, train_state, history_path, config.training, info_path)
