import os
import numpy as np
import torch
from models import model_factory
from utils import LoaderObject, get_test_data, normalization


#重加载模型并进行评估
class Evaluator:
    def __init__(self, config, example, base_name, output_dir):
        self.config = config
        self.example = example
        self.base_name = base_name
        self.output_dir = output_dir
        self.model = None
        self.loader = LoaderObject(output_dir, base_name)
        self.load_model()

    #加载模型格式与权重
    def load_model(self):
        data_config = self.config.data
        model_path = os.path.join(self.output_dir,
                                  f"{self.base_name}_model-{self.config.training.iterations}.pt")
        if not self.example.ics:
            data = self.example.get_data(data_config.num_domain, data_config.num_boundary,
                                         data_config.num_test)
        else:
            data = self.example.get_data(data_config.num_domain, data_config.num_boundary,
                                         data_config.num_test, data_config.num_initial)
        self.model = model_factory.create_model(self.config, data)
        if hasattr(self.config, "ipinn"):
            normalization(self.config, self.example, self.model)
        # 此处必须compile：建立 predict 所需的 outputs
        self.model.compile(self.config.training.optimizer, lr=self.config.training.lr)
        # map_location 强制加载到当前设备：支持 GPU 训练/CPU 评估等跨设备场景
        checkpoint = torch.load(model_path, weights_only=True,
                                map_location=torch.get_default_device())
        self.model.net.load_state_dict(checkpoint["model_state_dict"])

    #获取预测场数据和绝对误差数据
    def predict(self):
        x_test, u_true = get_test_data(self.example, self.config.data)
        u_pred = self.model.predict(x_test)
        abs_err = np.abs(u_pred - u_true)
        return x_test, u_pred, u_true, abs_err
        
    #获取按照固定时间顺序获取预测值（时间切片图用）
    def predict_slice(self, t, n_x=200):
        #传入非瞬态问题直接报错（稳态问题没有时间切片可分析）
        if self.example.time_domain is None:
            raise ValueError("predict_slice requires a time-dependent example")
        x_min, x_max = self.example.geom.l, self.example.geom.r
        x = np.linspace(x_min, x_max, n_x).reshape(-1, 1)
        xt = np.hstack([x, np.full_like(x, t)])
        u_pred = self.model.predict(xt)
        if self.example.exact_func is None:
            u_true = self.example.load_data(xt)
        else:
            u_true = self.example.u_exact_numpy(xt)
        return x.ravel(), u_pred.ravel(), u_true.ravel()

