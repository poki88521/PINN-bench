from deepxde import Model
import deepxde as dde
import numpy as np
import time
import torch
from trainers import StandardTrainer
from models import create_model


# ---------------------------------------------------------------- 输入归一化
class NormalizedNet(dde.nn.pytorch.NN):
    """输入归一化包装器：inner((x - mu) / sigma)。

    mu / sigma 由区域随机采样点统计得到（论文采样 1e5 点）。
    pde 中 dde.grad.jacobian / hessian 仍对原始坐标 x 求导，
    autograd 链式法则自动还原物理导数，无需改动 pde 本身。
    """

    def __init__(self, inner, mu, sigma):
        super().__init__()
        self.inner = inner
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "mu", torch.as_tensor(np.asarray(mu, dtype=np.float32), dtype=dtype)
        )
        self.register_buffer(
            "sigma", torch.as_tensor(np.asarray(sigma, dtype=np.float32), dtype=dtype)
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.get_default_dtype())
        return self.inner((x - self.mu) / self.sigma)


def compute_normalization(example, n_samples=100000):
    """采样区域点，返回输入归一化统计量 (mu, sigma)。"""
    geom = example.geomtime if example.time_domain is not None else example.geom
    X = geom.random_points(n_samples)
    sigma = X.std(0)
    sigma[sigma < 1e-8] = 1.0  # 防止某维无变化导致除零
    return X.mean(0), sigma


# ---------------------------------------------------------------- AW 自适应加权损失
def _make_aw_loss(sigma, gamma, log_scale):
    """构造论文 AW 损失：w(sigma) * L + log_scale * log(sigma^2 + 1/gamma)。

    deepxde 对每个误差分量单独调用 loss_fn(y_true, y_pred)，且 y_true/y_pred
    形状为 (N, 1)（并非多列拼接），因此不能用列切片区分 ic/bc/pde。
    log 正则项按该 sigma 分组的损失分量数均摊（log_scale = 1 / n_group），
    使总损失中 log 项总贡献恰为 log(sigma^2 + 1/gamma) 一次，与论文总损失一致。
    """

    def loss_fn(y_true, y_pred):
        loss = torch.mean(torch.square(y_true - y_pred))
        w = 1.0 / (sigma ** 2 + 1.0 / gamma)
        # 注意：sigma 为 dde.Variable（叶子张量），直接对张量表达式取 log，
        # 不要包 torch.tensor(...)——那会复制数值、断开与 sigma 的计算图。
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

    # 论文原项目：Adam + ExponentialLR(gamma=0.9)，但仅在 iter % 1000 == 0 时
    # 手动 scheduler.step()，等效于每 1000 次迭代衰减一次，即 deepxde 的
    # decay=("step", step_size, gamma)（torch.optim.lr_scheduler.StepLR）。
    # 注意 ("exponential", 0.9) 是每个 epoch 都乘 0.9，与原项目行为不符。
    # 可在 yaml training 段用 lr_decay_step / lr_decay_gamma 覆盖默认值。
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
    # 输入归一化：把内部网络包装成 (x - mu) / sigma 后再前向（训练/预测自动生效）
    mu, sigma = compute_normalization(example)
    model.net = NormalizedNet(model.net, mu, sigma)
    loss_history, train_state, model = train(model, config.training, model_path, example)
    StandardTrainer.test(example, model, config.data)
    StandardTrainer.save(loss_history, train_state, history_path, config.training, info_path, example)
