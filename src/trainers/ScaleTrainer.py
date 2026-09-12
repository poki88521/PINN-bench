import os
import time

import deepxde as dde
import numpy as np
import scipy.io as sio
import torch

from models import ScaleModel as scale
from trainers.monitors import ScaleMonitor
from utils import ScaleWriter


#Scale-PINN 训练器：纯 torch 训练循环（序贯修正），保留 launch() 以接入现有调度器
#超参全部取自配置的 scale 覆盖节，不再写死在代码里
class ScaleTrainer:
    def __init__(self, config, example, output_dir, base_name, base_config=None):
        self.config = config
        self.base_config = base_config if base_config is not None else config
        self.example = example
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, "csv")
        os.makedirs(self.csv_dir, exist_ok=True)
        self.base_name = base_name
        dims_config = config.dims
        self.dims = ([dims_config.in_dim] + [dims_config.width] * dims_config.depth
                     + [dims_config.out_dim])
        self.monitor = None
        self.writer = None
        self.pool = None
        self.model = None
        self.optimizer = None

    #启动：训练 -> 保存（与 TrainerObject.launch 的同名流程）
    def launch(self):
        self.monitor = self.make_monitor()
        self.writer = self.make_writer()
        loss_history, train_state = self.train(self.config.training)
        self.save(loss_history, train_state, self.config.training)

    #检测器：记录 l2 误差
    def make_monitor(self):
        return ScaleMonitor(self.example, self.config)

    #写入器：四个指标 csv（info 里额外含训练耗时）
    def make_writer(self):
        return ScaleWriter(self.csv_dir, self.base_name, self.monitor)

    #保存指标文件（列名与 std/ipinn 一致，可直接被 LoaderObject 读取）
    def save(self, loss_history, train_state, training_config):
        self.writer.history(loss_history, training_config)
        self.writer.components(loss_history, self.example)
        self.writer.info(train_state)
        self.writer.l2_error()

    #获取训练点池：复用算例数据集的参考解网格
    #列序：[x, t]；给出四个分量的索引与掩码（列掩码用于全池测试损失）
    def get_point_pool(self):
        dataset = sio.loadmat(self.example.dataset_path)
        x_axis = np.sort(np.ravel(dataset["x"]))
        t_axis = np.sort(np.ravel(dataset["t"]))
        #全网格坐标（[x, t]序），真值复用算例类的load_data
        xg, tg = np.meshgrid(x_axis, t_axis, indexing="ij")
        X = np.column_stack([xg.ravel(), tg.ravel()])
        Y = self.example.load_data(X)
        #分类：t=t_min为初始点，x=x_min/x_max为左右边界点，其余为内点
        m_ic = np.isclose(X[:, 1], t_axis[0])
        m_bc_l = np.isclose(X[:, 0], x_axis[0])
        m_bc_r = np.isclose(X[:, 0], x_axis[-1])
        m_in = ~(m_ic | m_bc_l | m_bc_r)
        dtype = torch.get_default_dtype()
        device = torch.get_default_device()
        pool = {
            "X": torch.as_tensor(X, dtype=dtype, device=device),
            "Y": torch.as_tensor(Y, dtype=dtype, device=device),
        }
        for name, mask in (("m_in", m_in), ("m_ic", m_ic),
                           ("m_bc_l", m_bc_l), ("m_bc_r", m_bc_r)):
            mask_t = torch.as_tensor(mask, device=device)
            pool[name] = mask_t
            pool["idx_" + name[2:]] = torch.nonzero(mask_t).squeeze(1)
        return pool

    #抽取一批训练点：内点/初始点/左右边界点分别采样后拼接
    #bs_in/bs_ic/bs_bc 为每步采样数（不是总点数），左右边界各取一半
    def get_batch(self, pool, bs_in, bs_ic, bs_bc):
        device = pool["X"].device
        #从给定索引集中抽 n 个（n<=0 返回空）
        def pick(idx, n):
            if n <= 0:
                return idx[:0]
            return idx[torch.randint(len(idx), (n,), device=device)]
        n_l = bs_bc // 2
        parts = [("m_in", pick(pool["idx_in"], bs_in)),
                 ("m_ic", pick(pool["idx_ic"], bs_ic)),
                 ("m_bc_l", pick(pool["idx_bc_l"], n_l)),
                 ("m_bc_r", pick(pool["idx_bc_r"], bs_bc - n_l))]
        sel = torch.cat([idx for _, idx in parts])
        batch = {"X": pool["X"][sel], "Y": pool["Y"][sel]}
        #按拼接顺序生成四个分量掩码
        offset = 0
        for (name, idx) in parts:
            mask = torch.zeros(len(sel), dtype=torch.bool, device=device)
            mask[offset:offset + len(idx)] = True
            batch[name] = mask
            offset += len(idx)
        return batch

    #PDE 残差（Allen-Cahn：u_t - d*u_xx - 5*(u - u^3)）
    def pde(self, forward_dict):
        u = forward_dict["u"]
        u_t = forward_dict["u_t"]
        u_xx = forward_dict["u_xx"]
        return u_t - self.config.example.d * u_xx - 5 * (u - u ** 3)

    #序贯修正项：(u_k-u_{k-1})/ER - γ*(u_xx_k-u_xx_{k-1})/ER_xx
    #γ 取物理扩散系数 d；ER/ER_xx 对应论文的 τ_sc/τ_α
    def correction(self, cur, u_pre, u_xx_pre):
        sc = self.config.scale
        gamma = self.config.example.d
        return ((cur["u"] - u_pre) / sc.ER
                - gamma * (cur["u_xx"] - u_xx_pre) / sc.ER_xx)

    #掩码均值（掩码为空时返回0，保持计算图连通）
    def masked_mean(self, value, mask):
        if not bool(mask.any()):
            return value.sum() * 0.0
        return value[mask].mean()

    #四个损失分量：PDE（含序贯修正）、左边界、右边界、初始条件
    #分量已乘权重，求和即总损失（与 ipinn 的 csv 记录口径一致）
    def loss_components(self, cur, u_pre, u_xx_pre, Y, batch):
        sc = self.config.scale
        residual = self.pde(cur) + self.correction(cur, u_pre, u_xx_pre)
        u = cur["u"]
        return [self.masked_mean(residual ** 2, batch["m_in"]),
                sc.weight_bc * self.masked_mean((u - Y) ** 2, batch["m_bc_l"]),
                sc.weight_bc * self.masked_mean((u - Y) ** 2, batch["m_bc_r"]),
                sc.weight_ic * self.masked_mean((u - Y) ** 2, batch["m_ic"])]

    #在全点池上计算四个损失分量（测试损失，不建参数梯度）
    def eval_loss_components(self, model, model_pre):
        pool = self.pool
        with torch.enable_grad():
            cur = model(pool["X"])
            pre = model_pre(pool["X"])
        comps = self.loss_components(cur, pre["u"].detach(), pre["u_xx"].detach(),
                                     pool["Y"], pool)
        return [float(c.detach()) for c in comps]

    #学习率调度：cosine 为 optax 余弦衰减（warmup=0 时等价于官方设置），其余为恒定
    #返回值务必是 Python float：numpy 标量写进 optimizer 会污染存档（weights_only 加载失败）
    def lr_at(self, step, training_config):
        peak = float(training_config.lr)
        sc = self.config.scale
        if getattr(sc, "lr_schedule", "cosine") != "cosine":
            return peak
        decay_steps = max(int(sc.lr_decay_steps), 1)
        alpha = float(sc.lr_end) / peak if peak > 0 else 0.0
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * min(step, decay_steps) / decay_steps)))
        return float(peak * ((1.0 - alpha) * cosine ** float(sc.lr_exponent) + alpha))

    #保存权重（文件名与 ipinn 一致：{base_name}_model-{step}.pt）
    def save_model(self, model, optimizer, step):
        path = os.path.join(self.output_dir, f"{self.base_name}_model-{step}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, path)
        print(f"saved model: {path}")

    def train(self, training_config):
        sc = self.config.scale
        iterations = int(training_config.iterations)
        display_every = int(training_config.display_every)
        if display_every <= 0 or iterations % display_every != 0:
            raise ValueError("display_every 必须为正且能整除 iterations"
                             "（writer 按等间隔重建 step 列）")
        #固定随机种子：权重初始化与每步采样均可复现
        seed = int(getattr(sc, "seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)
        #两个网络：当前权重 w_k 与上一轮权重 w_{k-1}（用于序贯修正）
        model = scale.ScaleNet(self.dims, self.config.model)
        model_pre = scale.ScaleNet(self.dims, self.config.model)
        model_pre.load_state_dict(model.state_dict())
        #上轮权重只做前向，冻结其参数梯度
        for p in model_pre.parameters():
            p.requires_grad_(False)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)
        self.optimizer = optimizer
        self.pool = self.get_point_pool()

        loss_history = dde.model.LossHistory()
        train_state = dde.model.TrainState()
        bs_in, bs_ic, bs_bc = sc.batch_domain, sc.batch_initial, sc.batch_boundary

        #记录一步：训练分量 + 全池测试分量 + l2 误差
        def record(step, train_comps):
            train_comps = [float(c.detach()) for c in train_comps]
            test_comps = self.eval_loss_components(model, model_pre)
            #l2 与 ipinn 使用同一测试网格，便于横向对比
            u_pred = model.predict_u(self.monitor.x_test_tensor)
            self.monitor.update(step, u_pred.cpu().numpy())
            loss_history.append(step, np.array(train_comps), np.array(test_comps),
                                np.array([self.monitor.l2_errors[-1]]))
            total_train, total_test = sum(train_comps), sum(test_comps)
            if total_train < train_state.best_loss_train:
                train_state.best_step = step
                train_state.best_loss_train = total_train
                train_state.best_loss_test = total_test

        time_start = time.time()
        step = 0
        while True:
            batch = self.get_batch(self.pool, bs_in, bs_ic, bs_bc)
            #当前权重前向：需要二阶导与反传
            cur = model(batch["X"])
            #上轮权重前向：不反传，输出随后 detach
            pre = model_pre(batch["X"])
            comps = self.loss_components(cur, pre["u"].detach(), pre["u_xx"].detach(),
                                         batch["Y"], batch)
            #先记录当前状态（step 为已完成的更新次数），再决定是否继续更新
            if step % display_every == 0:
                record(step, comps)
            if step >= iterations:
                break
            for group in optimizer.param_groups:
                group["lr"] = self.lr_at(step, training_config)
            loss = sum(comps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #权重快照：step 后同步，作为下一步的 w_{k-1}
            model_pre.load_state_dict(model.state_dict())
            step += 1

        time_elapsed = time.time() - time_start
        train_state.training_time = time_elapsed
        print(f"elapsed time: {time_elapsed:.2f}s")
        self.save_model(model, optimizer, iterations)
        return loss_history, train_state
