## 使用方法

训练（项目根目录）：
```
set PYTHONPATH=src
D:\anaconda\envs\pinnbench\python.exe src\main_scale.py -n AllenCahn1D -v scale
```
默认 40000 轮，产物在 runs/AllenCahn1D/scale/（csv 在同名 csv 子目录）。

画图（history/solution/slice 三张 png，并打印与 ipinn 的 RL2 对比）：
```
D:\anaconda\envs\pinnbench\python.exe src\plot_scale.py -n AllenCahn1D -v scale
```
ipinn 目录不同用 --ipinn_dir；冒烟测试加 --iterations 100 --display_every 10。

## 改动记录

### 1. 文件清单

新增：
- `src/main_scale.py`：scale 独立训练入口（`--iterations` / `--display_every` / `--seed` 可覆盖配置）
- `src/plot_scale.py`：训练与对比图脚本（三张 png + 控制台打印 RL2）
- `src/utils/ScaleEvaluator.py`：加载 scale 权重并预测（不接入 dde 的 Model 体系）
- `configs/scale/AllenCahn1D_scale.yaml`：scale 配置（沿用“顶层 std 基线 + 同名覆盖节”约定）

改写：
- `src/models/ScaleModel.py`：类名保持你的 `ScaleNet`；删掉 `last_model`；`compute_u` 用 `layers[1:-1]` 走 silu、`layers[-1]` 线性输出（去掉重复的输出层，且输出层无激活）；新增 `predict_u` 供推理；权重初始化改为配置驱动（`He uniform`，对齐官方 he_uniform）
- `src/trainers/ScaleTrainer.py`：**不再继承 `TrainerObject`**（其 `__init__` 会走 dde 的 `create_dataset`/`create_model`，`save` 依赖 monitor/`loss_history`，纯 torch 循环用不上且需给 `model_factory` 加分支）；超参全部搬进配置；损失拆成 4 个分量；用 dde 的 `LossHistory`/`TrainState` 造实例填值以复用现有 CSV 口径；新增耗时记录、权重存档、随机种子、余弦学习率

少量增补（只加不改）：
- `src/trainers/monitors.py`：新增 `ScaleMonitor`（不含 dde Callback/Model 的轻量 l2 记录器，提供 `.steps` / `.l2_errors`）
- `src/utils/csv_writer.py`：新增 `ScaleWriter`（`info.csv` 追加 `training_time_s`）
- `src/utils/__init__.py`：导出 `ScaleWriter`、`ScaleEvaluator`
- `src/models/__init__.py`：沿用你的 `from .ScaleModel import ScaleNet as ScalePINN`
- `src/models/model_factory.py`：仅撤掉我此前误加的一个空行，无功能改动

未改动：standard / ipinn 的全部逻辑与既有产物（`runs/Helmholtz2D/standard`、`runs/Heat1D/ipinn` 等均未覆盖）。

### 2. 数值依据（避免凭空取值）

- 修正项公式：论文 Eq.4a–4c，与官方 `ac/plot_benchmark_ac.ipynb` cell 2 一致
  `pde + (u - u_0)/ER - v2*(u_xx - u0_xx)/ER_xx`
- 记号对应：`ER` = τ_sc，`ER_xx` = τ_α，`γ` = 算例物理系数（本算例取 `d`，与官方 `v2` 同）
- 有出处（官方 AC）：`ER=0.45`、`ER_xx=1.5`、`weight_ic=weight_bc=100`、`lr=2e-3`、每步采样 `1000 / 50 / 50`（官方 `BS_ALL=1000`、`BS_BIC=50`）、`lr_end=1e-10`、`lr_exponent=1.2`、`lr_decay_steps=500000`
- 沿用 bench 现有配置以求与 ipinn 可比：`d=0.001`、dims `5×70`、`num_test=1000`、`iterations=40000`（与 ipinn 同轮数；官方为 500000）
- 滤波尺度是导出量：α² = ER·d/ER_xx = 0.45×0.001/1.5 = 3e-4 → **α = 0.01732**（论文 AC 的 α=1e-4 对应的是另一套设定）
- 我自定、无出处：`display_every=100`、`seed=0`

### 3. 已验证（笔记本 CPU，100 轮冒烟）

- `py_compile` 与全部模块导入通过；`trainer_factory` 分派路径亦跑通
- 100 轮（seed=0）：总损失 224.5 → 0.296（单调下降）；test 分量同步下降
- 产物齐全：`history/components/l2/info` 四个 csv（info 含 `training_time_s`）、`_model-100.pt`、三张 png（`_history.png` / `_solution.png` / `_solution_slice.png`）
- components 分量名与 `COMPONENT_NAMES["AllenCahn1D"]` 的 4 项对齐（PDE / 左 BC / 右 BC / IC）
- 固定种子后两次 100 轮的 `l2.csv` **逐字节一致**（复现性通过）
- 耗时：100 轮约 14–22s（含每 10 步一次全池测试评估）→ 40000 轮粗估**约 1.5–3 小时（笔记本 CPU）**，家用机 GPU 会明显更快；真实耗时会写入 `info.csv`

### 4. 结果预期与现状（重要）

- 现状（seed=0，100 轮）：RL2 = 1.092(step0) → 0.973(step10) → 1.147(step80) → **1.137(step100)**。
- 该规模下方的差异极大：未固定种子时同参数跑出 0.92 / 1.04 / 1.14。**所以 100 轮完全不能下结论**，必须跑到 40000 轮。
- l2 早期上升是这套设定的正常现象，有 ipinn 实测为证：ipinn 在 step=100 是 0.9125、step=200 是 1.1059、step=300 是 1.1188（同样先升），最终到 40000 轮才降到 0.1775。
- 判定标准：`scale < 0.1775`（ipinn 40000 轮 RL2）即达成“比 ipinn 好”。基线对照为 ipinn 自带 control（std，0.5173）。
- 论文报的 AC RL2=5.4e-5 **不可对标**：论文用周期边界 + α=1e-4 + 官方数据(201×512)，本复现用 Dirichlet + d=0.001 + bench 数据(101×201)。

### 5. 已知差异与未做的事（不隐瞒）

- **单种子**：官方用 5 个种子取均值；此处默认单次运行（seed=0）。由上一条可见方差明显，建议至少换 `--seed 10`、`--seed 20` 各跑一次再下结论。
- **未做 ER=0 消融**（按你的决定跳过）。
- **只做 AC 一个算例**，未扩展 Burgers 等。
- 学习率调度按 optax `cosine_decay_schedule`（warmup=0）的公式实现；本机未安装 optax，**未能逐值比对**。因 `lr_decay_steps=500000` 远大于 40000，该调度在 40000 轮内近似恒定 2e-3（第 40000 步约 1.96e-3）。
- `ScaleMonitor` 复用 ipinn 的测试网格（`dde.geometry...uniform_points`），运行时会出现 dde 的 `Warning: 1000 points required, but 1035 points sampled.`，**无害**，ipinn 同样如此。
- 角点（x=±1 且 t=0）同时计入 BC 与 IC，与官方用 `ic`/`bc` 掩码的行为一致，非 bug。
- 未实现 solution 图与对照组（ipinn）的并排对比；`solution/slice` 只用 scale 自己的权重。
- `plot_scale.py` 里 ipinn 目录是本机绝对路径，家用机若不存在会自动跳过对比曲线（会打印提示），不会报错。

### 6. 若结果不理想，可调的旋钮（按预期收益排序）

1. `--seed`：先排除单次运气（方差约 ±0.1）
2. `scale.batch_domain`：官方 1000，可试 2000–5000（更多内点对 PDE 项更稳）
3. `scale.lr` 与 `scale.lr_decay_steps`：改为 `lr_decay_steps=40000` 让余弦真正衰减到末期
4. `dims`：改 `depth=4, width=128` 贴近官方网络（当前 5×70 是为了与 ipinn 可比）
5. `scale.weight_ic`/`weight_bc`：当前 100，若 BC/IC 主导可下调
