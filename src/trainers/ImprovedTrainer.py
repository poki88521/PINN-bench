import deepxde as dde
import time
import os
import torch
from trainers import TrainerObject, StandardTrainer
from trainers.monitors import ImprovedMonitor
from utils import ImprovedWriter, normalization


class ImprovedTrainer(TrainerObject):
    def __init__(self, config, example, output_dir, base_name, base_config=None):
        super().__init__(config, example, output_dir, base_name, base_config)
        self.sigmas = [dde.Variable(1.0), dde.Variable(1.0), dde.Variable(1.0)]
        self.sigma_names = ["sigma_pde", "sigma_bc", "sigma_ic"]

    def preprocess(self):
        normalization(self.config, self.example, self.model)

    def make_monitor(self):
        return ImprovedMonitor(self.example, self.config, self.sigmas, self.sigma_names)

    def make_writer(self):
        return ImprovedWriter(self.csv_dir, self.base_name, self.monitor)

    def train(self, training_config):
        ipinn_config = self.config.ipinn
        gamma = ipinn_config.gamma
        sigma_pde, sigma_bc, sigma_ic = self.sigmas

        # deepxde data.PDE.losses 的分量顺序：PDE 残差在前，BC/IC 在后。
        # 当前各算例 pde 均返回单个残差张量，故 n_pde = 1。
        n_pde = 1
        n_bc = len(self.example.bcs)
        n_ic = len(self.example.ics)

        losses = [_make_aw_loss(sigma_pde, gamma, 1.0 / n_pde)]
        losses += [_make_aw_loss(sigma_bc, gamma, 1.0 / n_bc)] * n_bc
        if n_ic:
            losses += [_make_aw_loss(sigma_ic, gamma, 1.0 / n_ic)] * n_ic

        external_vars = [sigma_pde, sigma_bc]
        if n_ic:
            external_vars.append(sigma_ic)

        self.model.compile(training_config.optimizer, lr=training_config.lr,
                      decay=("step", training_config.lr_decay_step, training_config.lr_decay_gamma),
                      loss=losses,
                      external_trainable_variables=external_vars)
        time_start = time.time()
        loss_history, train_state = self.model.train(iterations=training_config.iterations,
                                                batch_size=training_config.batch_size,
                                                model_save_path=self.model_path,
                                                display_every=training_config.display_every,
                                                callbacks=[self.monitor])
        time_elapsed = time.time() - time_start
        print(f"elapsed time: {time_elapsed:.2f}s")
        return loss_history, train_state

    def save(self, loss_history, train_state, training_config):
        super().save(loss_history, train_state, training_config)
        self.writer.sigma()

    def after_train(self, loss_history, train_state, model):
        # 训练结束后跑一个standard版本作为对照组（基础训练参数不变），命名为control
        std_output_dir = os.path.join(self.output_dir, "control")
        os.makedirs(std_output_dir, exist_ok=True)
        std_base_name = f"{self.example.name}_control"
        std_trainer = StandardTrainer(self.base_config, self.example,
                                      std_output_dir, std_base_name)
        std_trainer.launch()


#自适应加权损失
def _make_aw_loss(sigma, gamma, log_scale):

    def loss_fn(y_true, y_pred):
        loss = torch.mean(torch.square(y_true - y_pred))
        w = 1.0 / (sigma ** 2 + 1.0 / gamma)
        return w * loss + log_scale * torch.log(sigma ** 2 + 1.0 / gamma)

    return loss_fn




