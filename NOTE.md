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
- 新版本工作流必须继承自standard工作流，通过固定的接口来自定义并修改std工作流中的内容
- std版本同时充当模板和运行的功能
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
- main.py / trainer_factory 接入 get_version_config（当前顶层 trainer_name 为 StandardTrainer，直接分派会错误走 std）
- ImprovedTrainer / trainer_factory 同步到 StandardTrainer 的类结构（当前 ImprovedTrainer 仍是模块函数风格）
- 图表制作功能（基础已做，扩展中）
- 设置自动std基准训练（每进行一次版本训练就自动跑一个std版本的模型作为对照）
- 新图表：
  - solution_plot（预测 vs 精确解，需模型重载 .pt）(可视化？)
- yaml模板更新
- **可选待办事项**
  - 保存最优模型（可以近似认为最后的模型就是最优模型）
  - 设置存档点（目前可以认为训练流程都较短因此不设置中断存档）
  - 图表中设置希腊字母（美观度问题，暂时忽略）
  - 文件名等代码风格重构（不影响代码使用）
  - 确认数据集存在否则报错（暂不影响使用）



## 保留问题
- l2_error 曲线缺 step=0 点：TrainMonitor 的 on_epoch_end 只在 `step % display_every == 0` 时记录，首个触发点是 display_every，因此 l2.csv 缺少起点 0（history.csv 从 0 开始）。若后续需要 l2 从 0 开始，可在 on_epoch_begin/on_train_begin 补记初始误差。
- path_init 的返回参数传递可优化：目前返回 version_dir/model_path/history_path 三元组，info/components/l2 路径均由 history_path 字符串替换推导，调用方需手动拼装。可考虑统一改为返回路径集合对象或字典，减少调用方对命名约定的依赖。
