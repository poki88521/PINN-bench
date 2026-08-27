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
- 函数前一行， 类的第一行写注释表示功能（此注释通常为一行，不得超过三行）

## 配置分层约定
- 顶层配置 = std 基线（FNN + StandardTrainer）；与 version 同名的覆盖节（如 ipinn:）= 该版本专属配置（AttentionNet + ImprovedTrainer + lr_decay + gamma + n_samples）
- utils/ConfigUtils.py 提供：AttrDict、load_yaml、merge_config(base, override)（递归覆盖合并，基于 deepcopy；已验证 deepcopy 保留 AttrDict 类型与点号访问）、get_version_config(config)（存在同名覆盖节则合并，否则返回顶层）
- trainer_factory 与 plot.py 均已接入 get_version_config：trainer 用合并后配置分派（顶层 trainer_name 是 StandardTrainer，覆盖节内才是 ImprovedTrainer）

## 代码结构（src/）
- models/：FNN（StandardModel.py）、AttentionNet（AttentionModel.py）、model_factory.create_model
- trainers/：TrainerObject（基类模板）、StandardTrainer、ImprovedTrainer、monitors.py、trainer_factory（均已同步为类结构）
- plotters/：StandardPlotter（history/components/l2/solution）、ImprovedPlotter（+sigma/compare 系列）、plotter_factory
- utils/：path_utils、config_utils、csv_writer（WriterObject/ImprovedWriter）、csv_loader（LoaderObject/ImprovedLoader）、Evaluator、other_utils
- plot.py：-n -v -o 参数，csv_dir = RUNS_DIR/config.version；main.py：训练入口

## Trainer 继承重写点（已落地为 TrainerObject）
- 基类 TrainerObject：__init__(config→data→model→preprocess) → launch(make_monitor→make_writer→train→save→after_train)
- 钩子（可重写）：preprocess（默认pass，ipinn做归一化）、make_monitor（默认TrainMonitor）、make_writer（默认WriterObject）、after_train（默认pass，ipinn跑std对照）
- 步骤（可重写但签名固定）：train(training_config, monitor)→(loss_history, train_state)、save(...)、create_dataset
- launch 禁止重写；train 返回类型固定；model_path 为属性
- 子类（ImprovedTrainer）重写：preprocess/make_monitor/make_writer/train/save(补sigma)/after_train(跑control)
- 对照组目录：runs/{example}/{version}/control/（与 ImprovedPlotter.loader_c 路径约定一致）

## 已确认结论
- l2 上升是训练长度问题（100 iter 太短），代码正确；重载 .pt 重算 l2 与 l2.csv 完全一致
- std 对照训练调度建议放版本 trainer 内部（ImprovedTrainer.launch 末尾），trainer_factory 只做配置合并+分派

## 待办事项
- readme更新（时间定于ipinn版本基本完成后）
  + 文件超链接？
  + 非win系统的启动方式
  + 对照实验说明：ipinn 训练自动跑 std 对照组（control 目录）+ 对比图（history_compare / components_compare / l2_compare）
  + 脚本说明补全：env.bat 环境引导、脚本可传参（run_std/run_ipinn/plot 的 -n/-v/-o）
  + 设备说明：代码已自动 cuda/cpu（main.py/plot.py 的 torch.set_default_device），GPU 训练/CPU 评估跨设备（Evaluator 的 map_location）
  + 新图表提及：solution_plot（热力图，三子图）/ solution_slice_plot（时间切片，5 时刻，非 time-dependent 自动跳过）
  + -o 参数具体命令行示例（如 scripts\plot.bat -v ipinn -o solution_plot）
  + examples.md 是否需同步更新（新增图表后）
- 版本分支

- **可选待办事项**
  - solution_plot 对照组对比（暂不做，solution暂时不需要与对照组比较）
  - solution_plot 可选动画形式（(x,y,t) 类算例逐帧合成）
  - CSV 读写类命名（WriterObject / LoaderObject 体系）可能再调整（等用户通知后统一改）
  - 保存最优模型（可以近似认为最后的模型就是最优模型）
  - 设置存档点（目前可以认为训练流程都较短因此不设置中断存档）
  - 图表中设置希腊字母（美观度问题，暂时忽略）
  - 文件名等代码风格重构（不影响代码使用）
  - 确认数据集存在否则报错（暂不影响使用）
  - 规则1（函数/类前一行注释 ≤3行）全面补全（用户自行按规则补齐）
  - 增加算例（要重构的内容可能比较多，先不加）
  - 自定义插值位置（会考虑插值误差的情况，较为复杂，先不加）
  - 把example_factory的字典优化掉（改成反射获取类名遍历对比）

- **新增发现（本次检查补充）**
  - main.py 创建 example 用的是原始 config，而 trainer_factory 内部分派用 merged 配置：目前 example 参数未被版本覆盖（example 节相同），若未来某版本覆盖 example 节参数会导致训练与评估不一致，需留意
  - Evaluator.load_model 用 `f"{base_name}_model-{iterations}.pt"` 硬编码最终迭代步：若训练中途中断或 iterations 改变，文件可能不存在；更稳妥是扫描 `model-*.pt` 取最大 step（latest checkpoint）
  - Evaluator.load_model 每次 compile 耗时约 3.5s（重建计算图），频繁调用评估/画图时可考虑缓存已编译模型
  - Burgers1D / AllenCahn1D 的 solution_plot 未实测（依赖 dataset 数据文件，已确认 dataset/ 下 Allen_Cahn.mat、Burgers.npz 存在，但画图链路未验证）



## 保留问题
- l2_error 曲线缺 step=0 点：TrainMonitor 的 on_epoch_end 只在 `step % display_every == 0` 时记录，首个触发点是 display_every，因此 l2.csv 缺少起点 0（history.csv 从 0 开始）。若后续需要 l2 从 0 开始，可在 on_epoch_begin/on_train_begin 补记初始误差。
- ImprovedPlotter.sigma_plot 未处理 CSVLoader.sigma() 返回 None 的情况：若 sigma 文件缺失（如被删、训练中断），解包 None 会崩。正常流程（ipinn 必有 sigma 文件）不触发，暂不处理，仅记录。

