import deepxde as dde
import numpy as np

from utils import get_test_data


#训练检测器（Callback的子类，其他Monitor的父类）
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


class ImprovedMonitor(TrainMonitor):
    def __init__(self, example, config, sigmas, sigma_names):
        super().__init__(example, config)
        self.sigmas = sigmas
        self.sigma_names = sigma_names
        self.sigma_steps = []
        self.sigma_values = {name: [] for name in sigma_names}

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.model.train_state.step % self.display_every == 0:
            self.sigma_steps.append(self.model.train_state.step)
            for name, sigma in zip(self.sigma_names, self.sigmas):
                self.sigma_values[name].append(sigma.item())

        