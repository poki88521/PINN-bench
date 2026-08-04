## PINN-bench
a benchmark that incorporates multiple example for PINN bases on deepXDE and PyTorch

## 功能
- 包含以下算例且可以通过.yaml的配置文件来修改超参数：
    + 一维热传导方程（Heat1D）
    + 一维波动方程（Wave1D）
    + Allen-Cahn方程（AllenCahn1D）
    + 一维Burgers方程（Burgers1D）
    + 二维Helmholtz方程（Helmholtz2D）
- 自动训练、打印、保存数据、制作评估图表

## 算例简介
见examples.md

## 快速开始
- 在项目根目录下运行main.py即可

## 项目结构
```text
PINN-Bench/
├── configs/          # 算例配置文件
├── runs/             # 训练输出（模型、日志、图片）
├── dataset/          # 无解析解的算例的数据集
├── src/
│   ├── examples/     # 各个算例类（继承 BaseProblem）
│   ├── models/       # 网络结构（FeedForwardNet 等）
│   ├── trainers/     # 训练流程（HelmholtzTrainer 等）
│   └── utils/        # 工具（配置加载、采样、绘图）
├── requirements.txt
└── README.md
```

## 注意事项
- 对于方程输入的x，稳态情况下（如Helmholtz2D），x表示两个空间维度，瞬态情况下（如Heat1D），x表示时间和空间两个维度

## 待办事项
- 从数据集中读取参考数据，并根据输入的数组来获取检测点
- Burgers1D
- 可供新网络、新训练方式进行拓展的接口（待设计）
- 保存最优模型（可选，可以近似认为最后的模型就是最优模型）