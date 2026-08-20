import time
import os
import numpy as np
from deepxde import Model
import deepxde as dde
from models import create_model
from utils import CSVUtils, get_test_data


class TrainMonitor(dde.callbacks.Callback):
    def __init__(self, example, config):
        super().__init__()
        self.x_test, self.u_true = get_test_data(example, config.data)
        self.display_every = config.training.display_every
        self.steps = []
        self.l2_errors = []

    def on_epoch_end(self):
        if self.model.train_state.step % self.display_every == 0:
            u_pred = self.model.predict(self.x_test)
            self.l2_errors.append(np.linalg.norm(self.u_true - u_pred) / np.linalg.norm(self.u_true))
            self.steps.append(self.model.train_state.step)


def path_init(config, example_dir):
    version_dir = os.path.join(example_dir, config.version)
    os.makedirs(version_dir, exist_ok=True)
    model_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_model")
    history_path = os.path.join(version_dir, f"{config.example.name}_{config.version}_history.csv")
    return version_dir, model_path, history_path


def train(model : Model, training_config, model_path, monitor):
    model.compile(training_config.optimizer, lr=training_config.lr)
    time_start = time.time()
    loss_history, train_state = model.train(iterations=training_config.iterations,
                                            batch_size=training_config.batch_size,
                                            model_save_path=model_path,
                                            display_every=training_config.display_every,
                                            callbacks=[monitor])
    time_elapsed = time.time() - time_start
    print(f"elapsed time: {time_elapsed:.2f}s")
    return loss_history, train_state, model


def create_dataset(example, data_config):
    if data_config.num_initial == 0:
        data = example.get_data(data_config.num_domain, data_config.num_boundary, data_config.num_test)
    else:
        data = example.get_data(data_config.num_domain, data_config.num_boundary,
                                data_config.num_test, data_config.num_initial)
    return data


def save(loss_history, train_state, history_path, training_config, example, monitor):
    CSVUtils.history_writer(history_path, loss_history, training_config)
    CSVUtils.info_writer(history_path, train_state)
    CSVUtils.component_writer(history_path, loss_history, example)
    CSVUtils.l2_error_writer(history_path, monitor)
    pass


def launch(config, example, example_dir):
    version_dir, model_path, history_path = path_init(config, example_dir)
    data = create_dataset(example, config.data)
    model = create_model(config, data)
    monitor = TrainMonitor(example, config)
    loss_history, train_state, model = train(model, config.training, model_path, monitor)
    save(loss_history, train_state, history_path, config.training, example, monitor)
