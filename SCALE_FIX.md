## 使用方法

训练/画图：
```
set PYTHONPATH=src
%PY%\src\main_scale.py -n AllenCahn1D -v scale
%PY%\src\plot_scale.py  -n AllenCahn1D -v scale
```
PY=D:\anaconda\envs\pinnbench\python.exe。默认 40000 轮，产物在 runs/AllenCahn1D/scale/（csv 在同名子目录）：三张 png + RL2 对比（含 ipinn），场图用 best 权重。覆盖项：--iterations/--display_every、--seed、--ER 0（关序贯修正，官方 ref 组）、--tag X（产物落 scale/X/，文件名加 _X，防覆盖；画图同步加）。ipinn 目录不同加 --ipinn_dir。

## 改动记录

### 1. 文件清单

新增：
- `src/main_scale.py`：scale 独立训练入口（`--iterations` / `--display_every` / `--seed` / `--ER` / `--tag` 可覆盖配置）
- `src/plot_scale.py`：训练与对比图脚本（三张 png + 控制台打印 RL2，支持 `--tag`）
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
- 记号对应：`ER` = τ_sc，`ER_xx` = τ_α，`γ`（修正项二阶差前面的系数）= 算例物理系数（本算例取 `d`，与官方 `v2` 同）
- **更正（2026-09-13 晚）**：本节此前写的“α² = ER·d/ER_xx → α = 0.01732”**是错的**。官方 AC 代码里二阶差项前面的系数就是 PDE 自身的扩散系数 `v2`（本复现为 `d`），没有“由 α 反推 α²”这一步，代码里也从未用到 α。`ER`/`ER_xx` 的对应关系本身不变。
- 有出处（官方 AC）：`ER=0.45`、`ER_xx=1.5`、`weight_ic=weight_bc=100`、`lr=2e-3`、每步采样 `1000 / 50 / 50`（官方 `BS_ALL=1000`、`BS_BIC=50`）、`lr_end=1e-10`、`lr_exponent=1.2`
- **`lr_decay_steps` 已由 500000 改为 40000**（= `iterations`）。官方语义是 `decay_steps = max_iter` 且真跑到 500000；本复现只跑到 40000/80000，照抄 500000 会让 lr 全程停在峰值（见第 4 节）。
- 沿用 bench 现有配置以求与 ipinn 可比：`d=0.001`、dims `5×70`、`num_test=1000`、`iterations=40000`（与 ipinn 同轮数；官方为 500000）
- 我自定、无出处：`display_every=100`、`seed=0`

### 3. 已验证（笔记本 CPU，100 轮冒烟）

- `py_compile` 与全部模块导入通过；`trainer_factory` 分派路径亦跑通
- 100 轮（seed=0）：总损失 224.5 → 0.296（单调下降）；test 分量同步下降；l2 1.0921 → 1.1371
- **加了进度打印后数值逐项不变**（2.2451e+02 / 5.8470e+00 / … / 2.9625e-01、l2 1.1371 与改动前一致），说明打印没有扰动随机数
- ER=0 分支（`--ER 0`）：控制台打印 `correction: disabled (ER<=0, official ref control)`，**无 nan/inf**，100 轮正常下降（224.5 → 0.775）
- best 存档：`{base}_best.pt` 正常写出（含 `step` / `best_loss_train` / `best_loss_test`）；`ScaleEvaluator` 优先读它并打印 `(best_step=...)`
- `--tag` 隔离生效：产物落到 `runs/<name>/<version>/<tag>/`，文件名带 `_<tag>`；`plot_scale.py --tag` 能对上路径，画图+RL2 对比正常
- 耗时：100 轮约 8.7–8.9s（≈87–89 ms/step，含每 20 步一次全池测试评估）。首次运行曾出现 184 ms/step，复跑后恢复正常，判定为机器负载抖动而非代码差异
- 40000 轮粗估**约 1 小时**（笔记本 CPU）；真实耗时会写入 `info.csv` 的 `training_time_s`

### 4. 学习率与震荡（重要，2026-09-13 晚补充）

- 修复前 `lr_decay_steps=500000`，40000 轮内 lr 几乎不衰减：lr(0)=2.000e-3、lr(40000)=**1.962e-3**（峰值 98%）
- 修复后（`lr_decay_steps=40000`）：lr(10000)=1.654e-3、lr(20000)=8.706e-4、lr(30000)=1.995e-4、lr(38000)=4.448e-6、lr(40000)=1.000e-10
- 官方结果（家用机，改前配置）：40000 轮 RL2 末值 0.005543、最好 0.004667@39200、耗时 629.8s；80000 轮末值 0.019549、最好 0.003286@42700、耗时 1326.3s
- 震荡的实测特征：loss 尖峰与 l2 尖峰同步出现（解真的被踢出去）；四个分量同时跳；几百步内自行回落；log10 损失自相关 lag1≈0.66、lag2≈0.36，**无支配性周期**（随机游走式抖动，非有节奏振荡）
- 主因判断：lr 未衰减 → Adam 的步长恒为 ~2e-3，而**修正项平方后含 (单步变化量)²**，lr 恒定时该项不可能归零，构成由 lr 决定的地板。旁证：ipinn（`decay(step,1000,0.9)`，40000 步衰减约 1/68）的 l2 在 399 个间隔里只上升 3 次，而 control 上升 124 次
- 80000 轮无收益：按万步分桶的 l2 中位数 0.046→0.021→0.023→0.012→0.015→0.014→0.015→0.012，后 40000 步只是平台随机游走

### 5. 结果预期与现状

- 判定标准：`scale < 0.1775`（ipinn 40000 轮 RL2）即“比 ipinn 好”。当前已达成：0.0055 比 0.1775 低约 32 倍（best 0.0047 低约 38 倍）
- **口径提醒**：ipinn 那条曲线在 399 个间隔里只有 3 次上升，其 40000 步值就是它的最小值（`best_step=40000`）；scale 的 40000 步值是“末步抽奖”。两者都用末步比较是公平的，但 scale 的末步方差明显更大
- **归因提醒**：与 ipinn 的差异**不能**归因到序贯修正项，因为 lr 调度、网络结构（AttentionNet+tanh vs sin+silu）、损失（自适应权重+log 正则 vs 固定权重）、训练框架（dde vs 纯 torch）四处都不同。**正确的归因对照是 `--ER 0`**（官方 notebook cell 5 用的就是 `main(ER=0, ...)`，5 个种子）——同一份代码关掉修正项。std/control 属第三、四种实现，只能当背景数字
- 论文报的 AC RL2=5.4e-5 **不可对标**：论文用周期边界 + α=1e-4 + 官方数据(201×512)，本复现用 Dirichlet + d=0.001 + bench 数据(101×201)。因此“复现到论文的数字”用这套设定无法验证，能验证的只是“方法在 bench 设定下优于基线”
- 表述上限建议：**“按论文方法实现，在 bench 的 AllenCahn 设定下 RL2 比 ipinn 基线低约 1.5 个数量级”**；要说到“验证了论文结论”，还需补 ER=0 同代码对照 + 多种子

### 6. 已知差异与未做的事（不隐瞒）

- **单种子**：官方用 5 个种子取均值；此处默认单次运行（seed=0）。同参数未固定种子时曾跑出 0.92/1.04/1.14，方差明显，建议至少补 `--seed 10`、`--seed 20`
- **未做 ER=0 消融**（代码已就绪，尚未运行）
- **只做 AC 一个算例**，未扩展 Burgers 等
- 学习率调度按 optax `cosine_decay_schedule`（warmup=0）的公式实现；本机未安装 optax，**未能逐值比对**
- `ScaleMonitor` 复用 ipinn 的测试网格（`dde.geometry...uniform_points`），运行时会出现 dde 的 `Warning: 1000 points required, but 1035 points sampled.`，**无害**，ipinn 同样如此
- 角点（x=±1 且 t=0）同时计入 BC 与 IC，与官方用 `ic`/`bc` 掩码的行为一致，非 bug
- 未实现 solution 图与 ipinn 的并排对比；`solution/slice` 只用 scale 自己的权重
- `plot_scale.py` 里 ipinn 目录是家用机的绝对路径，换机器若不存在会自动跳过对比曲线（打印提示），不报错
- `correction()` 在 ER<=0 时被跳过，但 `model_pre` 的前向仍在每步执行（与官方 `pred_0` 无条件计算一致），**未做省算优化**
- **每步的耗时只打印到控制台，没有写进 csv**（`info.csv` 里只有整轮总耗时 `training_time_s`）

### 7. 本轮（2026-09-13 晚）的改动

对应 `log.md` 里“下周可以做的事情”：

1. **`configs/scale/AllenCahn1D_scale.yaml`**：`lr_decay_steps: 500000` → `40000`，并加注释提醒“必须与 `training.iterations` 一致”（改跑 80000 时这里也要改，否则 40000 步之后 lr 被钉在 1e-10 空转）
2. **best 权重落盘**（`ScaleTrainer.save_best_model`）：训练中一旦刷新 best（判据仍为训练损失，与 `info.csv` 口径一致）就在内存里留一份 `state_dict` 快照，训练结束写入 `{base}_best.pt`（含 `step`/`best_loss_train`/`best_loss_test`，**不含 optimizer 状态**）。`ScaleEvaluator.resolve_model_path(prefer_best=True)` 优先读它。原 `_model-{iterations}.pt` 仍照旧保存
   - 动机：80000 轮那次 best=0.003286@42700，但存下来的是末步 0.019549，差 5.9 倍；两张热力图此前用的是偏了的权重
3. **ER=0 guard**（`ScaleTrainer.use_correction`）：`ER<=0` 时跳过修正项相加，避免 `(u-u_pre)/ER` 除零出 nan。与官方 `if (ER > 0)` 一致
4. **控制台进度打印**：训练前打印配置摘要与“修正项开/关”，之后每 `display_every` 步打印一行
   `[  step/iterations] train … test … l2 … best …@… elapsed …s`，训练结束打印总耗时与 ms/step、best 的 step 与损失。**此前确实没有任何进度输出**（只有结束时的保存信息），你的“以为电脑坏了”是代码的问题，不是你的错觉
5. **运行隔离（`--tag X`）**：`main_scale.py --tag` 把产物落到 `runs/<name>/<version>/X/` 且文件名加 `_X` 后缀；`plot_scale.py --tag` 同步。用于 ER=0、seed=10/20 等多次运行互不覆盖（此前是靠手工搬文件夹）
6. 控制台打印一律用英文，避免重定向到 GBK 日志时中文乱码

**尚未运行**：修改后的 40000 轮（含 ER=0 组、seed 组）。本节 1–4 项只做过 100 轮冒烟，未验证 40000 轮下的收敛形态。
