import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from examples import create_example
from utils import (load_yaml, path_init, get_version_config, LoaderObject,
                   ScaleEvaluator)


#参与对比的运行组：tag 由 <key>_s<seed> 拼出，color 为绘图配色
#顺序即各子图的横轴/行序：基线组在前，配对斜率图因此表示"加成修正后的变化"
GROUPS = [
    {"key": "ER0", "label": "ER=0", "color": "tab:blue", "seeds": [0, 10, 20]},
    {"key": "ER045", "label": "ER=0.45", "color": "tab:red", "seeds": [0, 10, 20]},
]

#同组内用线型区分种子
SEED_STYLES = {0: "-", 10: "--", 20: ":"}

#端值口径：取曲线尾部该比例的点求中位数（末步是单点抽奖，方差过大）
TAIL_FRACTION = 0.2

#时间切片位置
SLICES = [0.0, 0.25, 0.5, 0.75, 1.0]


def get_args():
    parser = argparse.ArgumentParser(description="compare scale runs across seeds")
    parser.add_argument("-n", "--name", default="AllenCahn1D", help="name of example")
    parser.add_argument("-v", "--version", default="scale", help="name of version")
    return parser.parse_args()


#尾部窗口中位数：端值的稳健口径
def tail_median(values, fraction=TAIL_FRACTION):
    arr = np.asarray(values, dtype=float)
    n_tail = max(1, int(round(len(arr) * fraction)))
    return float(np.median(arr[-n_tail:]))


#每组每次运行的定位信息与数据加载器（tag 目录即 main_scale.py --tag 的产物）
def collect_runs(output_dir, name, version):
    runs = []
    for group in GROUPS:
        for seed in group["seeds"]:
            tag = f"{group['key']}_s{seed}"
            base_name = f"{name}_{version}_{tag}"
            run_dir = os.path.join(output_dir, tag)
            csv_dir = os.path.join(run_dir, "csv")
            runs.append({"group": group, "seed": seed, "tag": tag,
                         "base_name": base_name, "run_dir": run_dir,
                         "csv_dir": csv_dir,
                         "loader": LoaderObject(csv_dir, base_name)})
    return runs


#读入每个运行的 l2 与损失曲线（文件缺失直接报错，避免静默画出空图）
def read_curves(runs):
    for run in runs:
        step, l2 = run["loader"].l2()
        run["l2_step"] = np.asarray(step, dtype=float)
        run["l2"] = np.asarray(l2, dtype=float)
        iteration, train_loss, test_loss = run["loader"].history()
        run["iteration"] = np.asarray(iteration, dtype=float)
        run["train_loss"] = np.asarray(train_loss, dtype=float)
        run["test_loss"] = np.asarray(test_loss, dtype=float)
        run["l2_tail"] = tail_median(run["l2"])
    return runs


#图1：l2 三面板（A 六条曲线叠加+组内包络 / B 端值散点 / C 配对斜率）
def plot_l2_panels(runs, out_path):
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0], wspace=0.28)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])

    #A：6 条 l2 曲线，颜色区分 ER 组、线型区分种子；同组加 min-max 包络看组内带宽
    for group in GROUPS:
        members = [r for r in runs if r["group"] is group]
        for run in members:
            ax_a.semilogy(run["l2_step"], np.clip(run["l2"], 1e-12, None),
                          color=group["color"], linestyle=SEED_STYLES[run["seed"]],
                          linewidth=1.3, label=f"{group['label']} seed {run['seed']}")
        if len({len(r["l2_step"]) for r in members}) == 1:
            stack = np.vstack([np.clip(r["l2"], 1e-12, None) for r in members])
            ax_a.fill_between(members[0]["l2_step"], stack.min(0), stack.max(0),
                              color=group["color"], alpha=0.15, linewidth=0)
        else:
            print(f"skip band for {group['label']}: step grids differ")
    ax_a.set_title("Relative L2 error (6 runs)")
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("relative L2 error")
    ax_a.legend(fontsize=8)

    #B：端值（尾段中位数）散点 + 组均值横线，直接比组内散布与组间差距
    for i, group in enumerate(GROUPS):
        vals = np.array([r["l2_tail"] for r in runs if r["group"] is group])
        xs = i + np.linspace(-0.18, 0.18, len(vals))
        ax_b.semilogy(xs, vals, "o", color=group["color"], markersize=8)
        ax_b.semilogy([i - 0.32, i + 0.32], [vals.mean()] * 2,
                      color="k", linewidth=1.5)
        ax_b.text(i, vals.mean(), f"  mean {vals.mean():.2e}", fontsize=8, va="bottom")
    ax_b.set_xticks(range(len(GROUPS)))
    ax_b.set_xticklabels([g["label"] for g in GROUPS])
    ax_b.set_title("End value (tail median)")
    ax_b.set_ylabel("relative L2 error")
    ax_b.grid(alpha=0.3)

    #C：配对斜率图，每个种子一条线从基线组连到修正组（同标尺下比种子噪声与方法效应）
    base_label, treat_label = GROUPS[0]["label"], GROUPS[-1]["label"]
    for seed in GROUPS[0]["seeds"]:
        vals = []
        for group in GROUPS:
            run = next(r for r in runs if r["group"] is group and r["seed"] == seed)
            vals.append(run["l2_tail"])
        ax_c.semilogy([0, 1], vals, marker="o", label=f"seed {seed}")
        ratio = vals[-1] / vals[0] if vals[0] > 0 else float("nan")
        print(f"seed {seed}: {base_label} {vals[0]:.6e} -> {treat_label} "
              f"{vals[-1]:.6e} (ratio {ratio:.3f})")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([g["label"] for g in GROUPS])
    ax_c.set_title("Paired by seed")
    ax_c.set_ylabel("relative L2 error")
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=8)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.12, wspace=0.28)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")


#图2：训练损失曲线（左 train / 右 test，各 6 条）
def plot_loss_panels(runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, key, title in ((axes[0], "train_loss", "Train loss"),
                           (axes[1], "test_loss", "Test loss")):
        for run in runs:
            ax.semilogy(run["iteration"], np.clip(run[key], 1e-12, None),
                        color=run["group"]["color"],
                        linestyle=SEED_STYLES[run["seed"]],
                        linewidth=1.3,
                        label=f"{run['group']['label']} seed {run['seed']}")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")


#加载 6 个权重并预测参考网格与时间切片（权重优先取 best 存档）
def predict_all(runs, config, example):
    x_axis, t_axis, X, u_true = ScaleEvaluator(config, example, runs[0]["base_name"],
                                               runs[0]["run_dir"]).reference_grid()
    for run in runs:
        evaluator = ScaleEvaluator(config, example, run["base_name"], run["run_dir"])
        evaluator.load_model()
        run["evaluator"] = evaluator
        run["u_pred"] = evaluator.predict_u(X)
        run["u_err"] = np.abs(run["u_pred"] - u_true)
        print(f"{run['tag']}: max |err| = {run['u_err'].max():.4e}")
    run_slices = []
    for t in SLICES:
        x_line, _, u_true_line = runs[0]["evaluator"].predict_slice(t)
        preds = [r["evaluator"].predict_slice(t)[1] for r in runs]
        run_slices.append((t, x_line, u_true_line, preds))
    return x_axis, t_axis, u_true, run_slices


#图3：误差热力图（2 行 ER 组 × 3 列种子，|Pred-Exact|，六格共用 log 色标）
def plot_error_heatmaps(runs, x_axis, t_axis, out_path):
    all_err = np.concatenate([r["u_err"] for r in runs])
    positive = all_err[all_err > 0]
    vmax = float(all_err.max())
    #色标下界取正误差的 1 分位并兜底，避免极小的单点值把色阶压扁
    vmin = float(np.percentile(positive, 1)) if positive.size else vmax * 1e-6
    vmin = max(vmin, vmax * 1e-8, 1e-16)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    print(f"error colour scale: vmin={vmin:.3e}, vmax={vmax:.3e} (shared, clipped below vmin)")
    Xg, Tg = np.meshgrid(x_axis, t_axis, indexing="ij")
    n_seed = max(len(g["seeds"]) for g in GROUPS)
    fig, axes = plt.subplots(len(GROUPS), n_seed,
                             figsize=(4.6 * n_seed, 4.2 * len(GROUPS)),
                             sharex=True, sharey=True, squeeze=False)
    im = None
    for row, group in enumerate(GROUPS):
        seed_vals = [r["u_err"].max() for r in runs if r["group"] is group]
        print(f"{group['label']}: max |err| among seeds = "
              f"{min(seed_vals):.3e} .. {max(seed_vals):.3e}")
        for col, seed in enumerate(group["seeds"]):
            run = next(r for r in runs if r["group"] is group and r["seed"] == seed)
            ax = axes[row, col]
            data = np.clip(run["u_err"], vmin, None).reshape(len(x_axis), len(t_axis))
            im = ax.pcolormesh(Xg, Tg, data, cmap="hot", shading="auto", norm=norm)
            ax.set_title(f"{group['label']} | seed {seed} | max {run['u_err'].max():.2e}",
                         fontsize=10)
            if col == 0:
                ax.set_ylabel("t")
            if row == len(GROUPS) - 1:
                ax.set_xlabel("x")
    fig.colorbar(im, ax=list(axes.ravel()), shrink=0.85, label="|Pred - Exact|")
    fig.suptitle("Abs error vs seed (shared log colour scale)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


#图4：时间切片对比（每个切片叠 6 条预测 + 精确解）
def plot_slices(runs, run_slices, out_path):
    fig, axes = plt.subplots(1, len(run_slices),
                             figsize=(4 * len(run_slices), 4), sharey=True)
    for ax, (t, x_line, u_true_line, preds) in zip(axes, run_slices):
        ax.plot(x_line, u_true_line, "k--", linewidth=2.0, label="exact")
        for run, u_pred in zip(runs, preds):
            ax.plot(x_line, u_pred, color=run["group"]["color"],
                    linestyle=SEED_STYLES[run["seed"]], linewidth=1.2,
                    label=f"{run['group']['label']} s{run['seed']}")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("u")
    fig.suptitle("Solution slices vs seed")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    args = get_args()
    torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path, output_dir, DATASET_DIR, base_name = path_init(BASE_DIR, args.name, args.version)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_yaml(config_path)
    merged = get_version_config(config)
    print(f"example name: {args.name}")
    print(f"version: {args.version}")
    print(f"device: {torch.get_default_device()}")
    print(f"read runs from: {output_dir}")
    example = create_example(args.name, DATASET_DIR, merged)

    print("=" * 60)
    runs = read_curves(collect_runs(output_dir, args.name, args.version))
    for run in runs:
        print(f"{run['tag']}: tail median L2 = {run['l2_tail']:.6e} "
              f"({len(run['l2'])} records)")
    print("=" * 60)

    prefix = os.path.join(output_dir, f"{args.name}_{args.version}_seeds")
    plot_l2_panels(runs, f"{prefix}_l2.png")
    plot_loss_panels(runs, f"{prefix}_loss.png")

    print("=" * 60)
    x_axis, t_axis, u_true, run_slices = predict_all(runs, merged, example)
    plot_error_heatmaps(runs, x_axis, t_axis, f"{prefix}_solution.png")
    plot_slices(runs, run_slices, f"{prefix}_slice.png")

    #控制台汇总：组内散布 vs 组间差距（种子影响的直接判据）
    print("=" * 60)
    means = {}
    for group in GROUPS:
        vals = np.array([r["l2_tail"] for r in runs if r["group"] is group])
        means[group["label"]] = vals.mean()
        within = vals.max() / vals.min() if vals.min() > 0 else float("nan")
        print(f"{group['label']}: tail median L2 mean={vals.mean():.6e}, "
              f"spread={vals.max() - vals.min():.6e} (within-group max/min {within:.3f})")
    base_label, treat_label = GROUPS[0]["label"], GROUPS[-1]["label"]
    if means[base_label] > 0:
        ratio = means[treat_label] / means[base_label]
        print(f"between-group ratio {treat_label}/{base_label} = {ratio:.3f} "
              f"(>1 worse than baseline)")
    print("done")
