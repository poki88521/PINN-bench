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
- 自动训练、打印、保存数据、制作评估图表
- 包含I-PINN为理论基础的改进结构代码（论文[2]）

## 算例简介
见examples.md

## 快速开始
- 在项目根目录下运行main.py即可

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
│   ├── utils/              # 工具包（YAML读取，绘图等）
│   └── main.py             # 主程序
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
4. 配置文件模板
```yaml
version: (version_name)
model_name: (model_name) #此处建议严格写模型类的名字，与model_factory中一致即可
trainer_name: (trainer_name) #此处建议严格写模型类的名字，与trainer_factory中一致即可
example:
  name: (example_name) #此处需要严格写算例类的名字
  a: 1.0 #仅为举例，此处写方程的参数（不包含边界条件、初始条件等）
training:
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


## 待办事项
- ~~对照试验进行的接口（每运行一次非standard版本代码自动运行一次standard版本作为对照组）~~修改为将I-PINN（Niu P, 2025）作为拓展模块的例子
- 保存最优模型（可选，可以近似认为最后的模型就是最优模型）
- 设置存档点（可选，目前可以认为训练流程都较短因此不设置中断存档）
- 图表制作功能（绘制损失曲线等）
- 有关本项目的环境配置方法





