import numpy as np
import torch


#获取测试集坐标与精确解
def get_test_data(example, data_config):
    if example.time_domain is None:
        x_test = example.geom.uniform_points(n=data_config.num_test, boundary=True)
    else:
        x_test = example.geomtime.uniform_points(n=data_config.num_test, boundary=True)
    if example.exact_func is None:
        u_true = example.load_data(x_test)
    else:
        u_true = example.u_exact_numpy(x_test)
    return x_test, u_true


#计算输入归一化的均值与标准差
def compute_normalization(example, n_samples=100000):
    geom = example.geomtime if example.time_domain is not None else example.geom
    X = geom.random_points(n_samples)
    sigma = X.std(0)
    sigma[sigma < 1e-8] = 1.0  # 防止某维无变化导致除零
    return X.mean(0), sigma


#给网络注册归一化 buffer 并挂 feature_transform（训练与重载模型共用）
def normalization(config, example, model):
    mu, sigma = compute_normalization(example, config.ipinn.n_samples)
    net = model.net
    net.register_buffer("norm_mu", torch.as_tensor(np.asarray(mu, dtype=np.float32),
                                                   dtype=torch.get_default_dtype()))
    net.register_buffer("norm_sigma", torch.as_tensor(np.asarray(sigma, dtype=np.float32),
                                                      dtype=torch.get_default_dtype()))
    net.apply_feature_transform(lambda x: (x - net.norm_mu) / net.norm_sigma)
