from deepxde import Model
import deepxde as dde
import numpy as np
import time
import torch
from trainers import StandardTrainer
from models import create_model


# ---------------------------------------------------------------- 输入归一化
def compute_normalization(example, n_samples=100000):
    geom = example.geomtime if example.time_domain is not None else example.geom
    X = geom.random_points(n_samples)
    sigma = X.std(0)
    sigma[sigma < 1e-8] = 1.0  # 防止某维无变化导致除零
    return X.mean(0), sigma


# ---------------------------------------------------------------- AW 自适应加权损失
def _make_aw_loss(sigma, gamma, log_scale):

    def loss_fn(y_true, y_pred):
        loss = torch.mean(torch.square(y_true - y_pred))
        w = 1.0 / (sigma ** 2 + 1.0 / gamma)
        return w * loss + log_scale * torch.log(sigma ** 2 + 1.0 / gamma)

    return loss_fn


def train(model: Model, training_config, model_path, example):
    gamma = 1000.0
    sigma_pde = dde.Variable(1.0)
    sigma_bc = dde.Variable(1.0)
    sigma_ic = dde.Variable(1.0)

    # deepxde data.PDE.losses 的分量顺序：PDE 残差在前，BC/IC 在后。
    # 当前各算例 pde 均返回单个残差张量，故 n_pde = 1。
    n_pde = 1
    n_bc = len(example.bcs)
    n_ic = len(example.ics)

    losses = [_make_aw_loss(sigma_pde, gamma, 1.0 / n_pde)]
    losses += [_make_aw_loss(sigma_bc, gamma, 1.0 / n_bc)] * n_bc
    if n_ic:
        losses += [_make_aw_loss(sigma_ic, gamma, 1.0 / n_ic)] * n_ic

    external_vars = [sigma_pde, sigma_bc]
    if n_ic:
        external_vars.append(sigma_ic)

    lr_decay_step = getattr(training_config, "lr_decay_step", 1000)
    lr_decay_gamma = getattr(training_config, "lr_decay_gamma", 0.9)

    model.compile(training_config.optimizer, lr=training_config.lr,
                  decay=("step", lr_decay_step, lr_decay_gamma), loss=losses,
                  external_trainable_variables=external_vars)
    time_start = time.time()
    loss_history, train_state = model.train(iterations=training_config.iterations,
                                            batch_size=training_config.batch_size,
                                            model_save_path=model_path,
                                            display_every=training_config.display_every)
    time_elapsed = time.time() - time_start
    print(f"elapsed time: {time_elapsed:.2f}s")
    return loss_history, train_state, model


def launch(config, example, example_dir):
    version_dir, model_path, history_path, info_path = StandardTrainer.path_init(config, example_dir)
    data = StandardTrainer.create_dataset(example, config.data)
    model = create_model(config.model_name, config.dims, data)

    mu, sigma = compute_normalization(example)
    net = model.net
    net.register_buffer("norm_mu", torch.as_tensor(np.asarray(mu, dtype=np.float32),
                                                   dtype=torch.get_default_dtype()))
    net.register_buffer("norm_sigma", torch.as_tensor(np.asarray(sigma, dtype=np.float32),
                                                      dtype=torch.get_default_dtype()))

    def normalize(x):
        return (x - net.norm_mu) / net.norm_sigma

    net.apply_feature_transform(normalize)
    loss_history, train_state, model = train(model, config.training, model_path, example)
    StandardTrainer.test(example, model, config.data)
    StandardTrainer.save(loss_history, train_state, history_path, config.training, info_path, example)
