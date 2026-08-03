## PINN-bench
a benchmark that incorporates multiple example for PINN bases on deepXDE and PyTorch

## 1.功能
- 包含以下算例且可以通过.yaml的配置文件来修改超参数：
    + 一维热传导方程（Heat1D）
    + 一维波动方程（Wave1D）
    + Allen-Cahn方程（AllenCahn1D）
    + 一维Burgers方程（Burgers1D）
    + 二维Helmholtz方程（Helmholtz2D）
- 自动训练、打印、保存数据、制作评估图表

## 2.快速开始
- 在项目根目录下运行main.py即可

## 3.项目结构
```text
PINN-Bench/
├── configs/          # 算例配置文件
├── runs/             # 训练输出（模型、日志、图片）
├── src/
│   ├── examples/     # 各个算例类（继承 BaseProblem）
│   ├── models/       # 网络结构（FeedForwardNet 等）
│   ├── trainers/     # 训练流程（HelmholtzTrainer 等）
│   └── utils/        # 工具（配置加载、采样、绘图）
├── requirements.txt
└── README.md
```

## 4.待办事项
- Heat1D
- Wave1D
- AllenCahn1D
- Burgers1D
- 可供新网络、新训练方式进行拓展的接口（待设计）
- 保存最优模型（可选，可以近似认为最后的模型就是最优模型）