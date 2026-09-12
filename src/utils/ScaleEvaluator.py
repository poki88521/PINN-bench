import glob
import os
import re

import numpy as np
import scipy.io as sio
import torch

from models import ScaleModel


#Scale-PINN 评估器：加载纯 torch 存档并预测（不接入 deepxde 的 Model 体系）
class ScaleEvaluator:
    def __init__(self, config, example, base_name, output_dir):
        self.config = config
        self.example = example
        self.base_name = base_name
        self.output_dir = output_dir
        dims_config = config.dims
        self.dims = ([dims_config.in_dim] + [dims_config.width] * dims_config.depth
                     + [dims_config.out_dim])
        self.model = None
        self.model_path = None

    #选取存档：优先配置指定的迭代步，否则取 step 最大的一个
    def resolve_model_path(self):
        want = os.path.join(self.output_dir,
                            f"{self.base_name}_model-{self.config.training.iterations}.pt")
        if os.path.isfile(want):
            return want
        pattern = os.path.join(self.output_dir, f"{self.base_name}_model-*.pt")
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"no scale checkpoint found: {pattern}")

        def step_of(path):
            match = re.search(r"_model-(\d+)\.pt$", path)
            return int(match.group(1)) if match else -1

        return max(candidates, key=step_of)

    #加载模型与权重
    def load_model(self):
        path = self.resolve_model_path()
        model = ScaleModel.ScaleNet(self.dims, self.config.model)
        checkpoint = torch.load(path, weights_only=True,
                                map_location=torch.get_default_device())
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        self.model_path = path
        print(f"loaded model: {path}")
        return path

    #批量预测 u（分块，避免一次性占用过多内存）
    def predict_u(self, X, chunk=20000):
        dtype = torch.get_default_dtype()
        device = torch.get_default_device()
        X = np.asarray(X, dtype=np.float64)
        out = []
        for start in range(0, len(X), chunk):
            part = torch.as_tensor(X[start:start + chunk], dtype=dtype, device=device)
            out.append(self.model.predict_u(part).cpu().numpy())
        return np.concatenate(out, axis=0)

    #参考解网格（来自算例数据集的 x/t 轴），返回按 [x, t] 展开的坐标与真值
    def reference_grid(self):
        dataset = sio.loadmat(self.example.dataset_path)
        x_axis = np.sort(np.ravel(dataset["x"]))
        t_axis = np.sort(np.ravel(dataset["t"]))
        xg, tg = np.meshgrid(x_axis, t_axis, indexing="ij")
        X = np.column_stack([xg.ravel(), tg.ravel()])
        u_true = self.example.load_data(X)
        return x_axis, t_axis, X, u_true

    #时间切片预测（固定 t 上的 x 分布）
    def predict_slice(self, t, n_x=200):
        x_min, x_max = self.example.geom.l, self.example.geom.r
        x = np.linspace(x_min, x_max, n_x).reshape(-1, 1)
        xt = np.hstack([x, np.full_like(x, t)])
        u_pred = self.predict_u(xt)
        u_true = self.example.load_data(xt)
        return x.ravel(), u_pred.ravel(), u_true.ravel()
