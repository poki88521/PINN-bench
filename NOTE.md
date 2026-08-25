## 简介
本文件主要内容为个人开发笔记

## 环境与运行
- 专用环境：D:\anaconda\envs\pinnbench\python.exe（python 3.11.15，torch 2.13.0+cpu，deepxde 1.15.0，matplotlib 3.11.0 渲染正常）
- 原 deepxde 环境的 matplotlib 因 numpy ABI 崩溃已弃用，本项目统一用 pinnbench 环境
- 运行训练：cd D:\code\PINN-Bench && set PYTHONPATH=src && D:\anaconda\envs\pinnbench\python.exe src\main.py -n 算例名 -v 版本名
- 输出重定向建议用 cmd（PowerShell 中文路径/编码问题多），或 base64+stdin 传 python 脚本

## 开发原则
- 整个项目为PINN改进尽可能提供模板和接口（即工作节内部被封装，改进时无需修改）
- 一个版本对应一个训练-评估与保存-绘图的工作流，训练入口应当和绘图入口分离，版本间尽可能分离
- 新版本工作流必须继承自standard工作流（或标准模板），通过固定的接口来实现新工作流中的内容
- std版本主要充当运行的功能，模板可能为独立内容
- 算例、模型、训练和绘图组件均通过factory进行调度

## 配置分层约定
- 顶层配置 = std 基线（FNN + StandardTrainer）；与 version 同名的覆盖节（如 ipinn:）= 该版本专属配置（AttentionNet + ImprovedTrainer + lr_decay + gamma + n_samples）
- utils/ConfigUtils.py 提供：AttrDict、load_yaml、merge_config(base, override)（递归覆盖合并，基于 deepcopy；已验证 deepcopy 保留 AttrDict 类型与点号访问）、get_version_config(config)（存在同名覆盖节则合并，否则返回顶层）
- 注意：main.py / trainer_factory 尚未接入 get_version_config，顶层 trainer_name 现为 StandardTrainer，直接分派会错误走 std（见待办）

## 代码结构（src/）
- models/create_model：按 config 切换 FNN / AttentionNet
- trainers/：StandardTrainer（类）、ImprovedTrainer（仍是旧模块函数风格，未同步到类结构）、monitors.py（TrainMonitor / ImprovedMonitor）、trainer_factory（未同步）
- plotters/：StandardPlotter（类，_plot 结尾方法自动收集，resolve_plots 支持 only 过滤）、ImprovedPlotter(StandardPlotter)（sigma_plot）、plotter_factory.PLOTTERS
- plot.py：-n -v -o 参数，csv_dir = RUNS_DIR/config.version

## Trainer 继承重写点（分析结论）
- create_dataset / save：不用重写（save 已用 hasattr(monitor, "sigma_names") 兼容 sigma 差异）
- train：必重写（ipinn 差异：AW loss 列表 + decay=("step",...) + external_trainable_variables；签名与返回值须与父类一致）
- __init__：建议不改；父类应加空钩子 preprocess()（建 model 后调用），ImprovedTrainer 重写它做 normalization（apply_feature_transform 必须在 compile 前挂到 net）
- launch：差异集中在 monitor 类型（TrainMonitor → ImprovedMonitor，需要 sigmas）与训练前预处理；建议父类抽 make_monitor() 钩子，或子类整体重写 launch
- path_init：建议重写或参数化（csv 文件名带 _name 占位符遗留；ipinn 末尾跑 std 对照需第二组路径）
- 最小改动方案：StandardTrainer 增加 preprocess() + make_monitor() 两个空钩子，ImprovedTrainer 继承并只重写 preprocess / make_monitor / train

## 已确认结论
- l2 上升是训练长度问题（100 iter 太短），代码正确；重载 .pt 重算 l2 与 l2.csv 完全一致
- std 对照训练调度建议放版本 trainer 内部（ImprovedTrainer.launch 末尾），trainer_factory 只做配置合并+分派

## 待办事项
- 新图表：
  - solution_plot（预测 vs 精确解，需模型重载 .pt）两种呈现形式均保留（scratch/solution_plot_demo.py 有示意）：
    - 形式A：场热力图三子图（预测 | 精确 | 绝对误差），通用性强，所有算例统一
    - 形式B：时间切片曲线（固定若干 t 画 u(x)），仅 time-dependent 算例适用
    - 扩展预案：若未来有 >2D 输入算例，热力图退化为切片热力图（固定第 3 维取若干值画 2D 场组图），4D+ 仅切片
  - 更多新模型与std模型的对照图（位于improved plotter中）
- yaml模板更新
- 脚本保存目录

- **可选待办事项**
  - solution_plot 可选动画形式（(x,y,t) 类算例逐帧合成）
  - CSV 读写类命名（WriterObject / LoaderObject 体系）可能再调整（等用户通知后统一改）
  - 保存最优模型（可以近似认为最后的模型就是最优模型）
  - 设置存档点（目前可以认为训练流程都较短因此不设置中断存档）
  - 图表中设置希腊字母（美观度问题，暂时忽略）
  - 文件名等代码风格重构（不影响代码使用）
  - 确认数据集存在否则报错（暂不影响使用）



## 保留问题
- l2_error 曲线缺 step=0 点：TrainMonitor 的 on_epoch_end 只在 `step % display_every == 0` 时记录，首个触发点是 display_every，因此 l2.csv 缺少起点 0（history.csv 从 0 开始）。若后续需要 l2 从 0 开始，可在 on_epoch_begin/on_train_begin 补记初始误差。
- ImprovedPlotter.sigma_plot 未处理 CSVLoader.sigma() 返回 None 的情况：若 sigma 文件缺失（如被删、训练中断），解包 None 会崩。正常流程（ipinn 必有 sigma 文件）不触发，暂不处理，仅记录。

