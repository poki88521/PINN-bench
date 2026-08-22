## PINN-bench
a benchmark that incorporates multiple example for PINN bases on deepXDE and PyTorch

## 功能
- 可以PINN改进的验证工具，为拓展和改进网络结构、训练流程、损失函数提供接口（详见使用方法）
- 包含以下算例且可以通过.yaml的配置文件来修改超参数：
    + 一维热传导方程（Heat1D）
    + 一维波动方程（Wave1D）
    + Allen-Cahn方程（AllenCahn1D）
    + 一维Burgers方程（Burgers1D）
    + 二维Helmholtz方程（Helmholtz2D）
- 自动训练、打印、保存数据
- 可以通过绘图入口自动画图（支持自定义图表，详见使用方法）
- 包含I-PINN为理论基础的改进结构代码（论文[2]）

## 算例简介
见examples.md

## 快速开始（Windows）
1. 环境配置

基础依赖由 conda 安装（避免 numpy 与 matplotlib 等二进制扩展的版本冲突），deepxde 用 pip 安装。

```bash
conda create -n pinnbench python=3.11 numpy scipy pandas matplotlib pyyaml scikit-learn -y
conda activate pinnbench
pip install deepxde torchinfo
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

说明：pytorch 为 CPU 版（无需 GPU 即可运行全部算例；如日后在带 NVIDIA 显卡的机器上运行，可改用 `pip install torch --index-url https://download.pytorch.org/whl/cu121`）

2. 启动：在项目根目录下运行以下命令即可：
```bash
python src/main.py -n Helmholtz2D -v standard
```

- `-n / --name`：算例名（Heat1D / Helmholtz2D / Wave1D / AllenCahn1D / Burgers1D）
- `-v / --version`：版本名，对应 `configs/{version}/` 目录下的配置文件（如 standard、ipinn）

不带参数时默认运行 `Helmholtz2D / standard`（该算例有解析解，不依赖 dataset 目录下的数据文件）
3. 绘图：......

## 项目结构
```text
PINN-Bench/
├── configs/                # YAML配置（命名格式为{exmaple}_{version}.yaml）
├── dataset/                # 算例数据集（无解析解时）
├── runs/                   # 运行结果（不同版本的模型文件、历史记录等）
|   └── (example_name)/            
|       ├── standard/       
|       └── (version_name)/             
├── src/                    # 代码区
│   ├── examples/           # 算例包（算例方程、边界条件等）
│   ├── models/             # 模型包（FNN模型）
│   ├── trainers/           # 训练包（完整训练流程）
│   ├── plotters/           # 绘图包（损失曲线、损失分量曲线）
│   ├── utils/              # 工具包（YAML读取，路径读取等）
│   ├── plot.py             # 绘图主程序
│   └── main.py             # 训练主程序
├── examples.md             # 算例说明
└── README.md
```
本项目中的模型、算例和训练器均已解耦，可以通过factory的方式进行拓展，详情见使用方法

## 使用方法
使用方法主要针对基础算例流程之外的修改方法
1. 算例
    - 算例方程中的参数等信息均可通过配置文件来进行修改（暂不支持复杂的边界条件与初始条件通过配置文件直接修改）
2. 模型
    - 网络大小可以通过修改配置文件直接修改
    - 不同种类的模型可以通过在models文件夹额外添加文件来构建，并在model_factory.py文件中添加新的模型实例构建代码
3. 训练
    - 训练迭代次数、批量大小等数据均可通过配置文件直接修改（不会再显示于输出中）
    - 和模型相同，不同类型的训练代码也可自定义文件并在trainer_factory.py中添加对应代码
    - 有其他需要在训练过程中记录的数据可以写Callback的子类，详见[dde官方文档](https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#module-deepxde.callbacks)
    - 在monitors.py中可以自定义Monitor，继承TrainMonitor即可
4. 评估与绘图
    - 绘图程序默认绘制所选版本的所有图表
    - 绘图程序也可自定义文件并在plotter_factory.py中添加代码
    - 通过传入`--only`参数，可以选择绘制哪些图表，但此处的参数必须与方法名**完全一致**，对照字典如下：
```python
plot_func_dict = {
  #standard
  "history_plot": history_plot,       #总损失-迭代数曲线
  "components_plot": components_plot,  #各个分量的损失-迭代数曲线
  #ipinn
  "sigma_plot": sigma_plot #损失函数权重-迭代数曲线
}
```

5. 配置文件模板
```yaml
version: (version_name) #版本名，需与configs下的子目录名一致

# 顶层为当前版本的 std 基线配置（也是 std 对照实验使用的配置）
example:
  name: "Helmholtz2D" #此处需要严格写算例类的名字
  a: 1.0 #仅为举例，此处写方程的参数（不包含边界条件、初始条件等）

model:
  model_name: (model_name) #此处建议严格写模型类的名字，与model_factory中一致即可
  activation: "tanh" 
  init: "Glorot normal" #激活函数和初始化方法需严格按照字典的名称写法（详见注意事项）

training:
  trainer_name: (trainer_name) #此处建议严格写训练器的名字，与model_factory中一致即可
  optimizer: "adam" #要与deepXDE的优化器名称匹配
  lr: 0.001
  iterations: 100
  batch_size: 128
  display_every: 10 #在控制台打印信息的迭代数间隔

data:
  num_domain: 10000 #区域内采样点
  num_boundary: 1000 #边界采样点
  num_initial: 500 #初始采样点（稳态问题必须设置为0占位！）
  num_test: 1000 #测试点数量

dims: #网络结构，包含输入输出维度、深度和宽度
  in_dim: 2
  depth: 5
  width: 70
  out_dim: 1

plots:
  history_plot: true
  components_plot: true #是否绘制此图表

(version_name): #与version同名的覆盖节，定义本版本相对std基线的差异（无则省略）
  model:
    model_name: (model_name) #覆盖顶层model.model_name
  training:
    trainer_name: (trainer_name) #覆盖顶层training.trainer_name
    lr_decay_step: 1000 #覆盖层可追加版本专属参数
    lr_decay_gamma: 0.9
  gamma: 1000.0 #版本专属参数（如ipinn的AW权重）
  n_samples: 100000

# 运行时：get_version_config(config) 将顶层与(version_name)节递归合并，
# 得到本版本有效配置；std对照直接使用顶层配置
```

## 注意事项
- 项目理论基础均来自论文[1]，拓展示例来自[2]
- 配置文件中的初始化名和激活函数名必须严格按照以下字典（来自deepxde和torch的代码）中的名字来设置（否则会读取不出来！）
```python
initializer_dict = {
  "Glorot normal": torch.nn.init.xavier_normal_,
  "Glorot uniform": torch.nn.init.xavier_uniform_,
  "He normal": torch.nn.init.kaiming_normal_,
  "He uniform": torch.nn.init.kaiming_uniform_,
  "zeros": torch.nn.init.zeros_,
}
activation_dict = {
"elu": bkd.elu,
"gelu": bkd.gelu,
"relu": bkd.relu,
"selu": bkd.selu,
"sigmoid": bkd.sigmoid,
"silu": bkd.silu,
"sin": bkd.sin,
"swish": bkd.silu,
"tanh": bkd.tanh,
}
```
- 对于方程输入的x，稳态情况下（如Helmholtz2D），x表示两个空间维度，瞬态情况下（如Heat1D），x表示时间和空间两个维度
- 测试部分使用`geom.uniform_points()`生成均匀测试点而非`data.test_x`
- 稳态问题**必须**在配置文件的`data.num_initial`栏目中设置0作为占位
- 采样时采样数量会补全到二进制整数，可忽略
- yaml读取到的字典已被工具打包为对象，可以通过调用对象中内容的方式来获取配置信息

## 论文引用
[1] Raissi, M., Perdikaris, P., Karniadakis, G.E., 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
[2] Niu, P., Guo, J., Chen, Y., Zhou, Y., Feng, M., Shi, Y., 2025. Improved physics-informed neural network in mitigating gradient-related failures. Neurocomputing 638, 130167. https://doi.org/10.1016/j.neucom.2025.130167








