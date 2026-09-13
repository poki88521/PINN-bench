import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples import create_example
from utils import (load_yaml, path_init, get_version_config, LoaderObject,
                   ScaleEvaluator)


#ipinn 历史结果默认目录（家用机产物副本，可用 --ipinn_dir 覆盖）
IPINN_DIR = (r"C:\Users\mzr\Desktop\report\reproduction\pinns-reproduction"
             r"\2.Improved physics-informed neural network in mitigating gradient-related failures"
             r"\runs\AllenCahn1D\ipinn")


def get_args():
    parser = argparse.ArgumentParser(description="draw Scale-PINN plots")
    parser.add_argument("-n", "--name", default="AllenCahn1D", help="name of example")
    parser.add_argument("-v", "--version", default="scale", help="name of version")
    parser.add_argument("--ipinn_dir", default=IPINN_DIR,
                        help="ipinn 历史结果目录，缺失则跳过对比曲线")
    parser.add_argument("--tag", default=None,
                        help="与 main_scale.py 相同的运行标记（读 runs/<name>/<version>/<tag>/）")
    return parser.parse_args()


#安全读取某条曲线：文件缺失返回 None
def try_read(loader, method):
    try:
        return getattr(loader, method)()
    except FileNotFoundError:
        return None


#左图：scale 自身损失历史；右图：l2 误差对比（scale vs ipinn vs ipinn 的 std 对照组）
def draw_history(output_dir, base_name, csv_dir, name, ipinn_dir):
    hist_path = os.path.join(csv_dir, f"{base_name}_history.csv")
    if not os.path.isfile(hist_path):
        raise FileNotFoundError(f"scale history not found: {hist_path}（先跑 main_scale.py）")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    iteration, train_loss, test_loss = LoaderObject(csv_dir, base_name).history()
    axes[0].set_title("Scale-PINN total loss")
    axes[0].semilogy(iteration, np.clip(train_loss, 1e-12, None), label="train")
    axes[0].semilogy(iteration, np.clip(test_loss, 1e-12, None), label="test")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    entries = [("scale", csv_dir, base_name, "tab:red")]
    if ipinn_dir and os.path.isdir(ipinn_dir):
        entries.append(("ipinn", ipinn_dir, f"{name}_ipinn", "tab:blue"))
        entries.append(("ipinn control (std)", os.path.join(ipinn_dir, "control"),
                        f"{name}_control", "tab:green"))
    else:
        print(f"skip ipinn curves (dir not found): {ipinn_dir}")

    axes[1].set_title("Relative L2 error")
    final_l2 = {}
    for label, directory, file_base, color in entries:
        got = try_read(LoaderObject(directory, file_base), "l2")
        if got is None:
            print(f"skip l2 curve (not found): "
                  f"{os.path.join(directory, file_base + '_l2.csv')}")
            continue
        step, l2 = got
        step, l2 = np.asarray(step), np.asarray(l2)
        axes[1].semilogy(step, l2, label=label, color=color)
        final_l2[label] = float(l2[-1])
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative L2 error")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{base_name}_history.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")
    return final_l2


#预测场热力图（预测 | 参考解 | 绝对误差），网格取自算例数据集的 x/t 轴
def draw_solution(output_dir, base_name, evaluator, name):
    x_axis, t_axis, X, u_true = evaluator.reference_grid()
    u_pred = evaluator.predict_u(X)
    nx, nt = len(x_axis), len(t_axis)
    Xg, Tg = np.meshgrid(x_axis, t_axis, indexing="ij")
    U = u_pred.reshape(nx, nt)
    T = u_true.reshape(nx, nt)
    E = np.abs(U - T)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, data, title, cmap in ((axes[0], U, "Predicted", "RdBu_r"),
                                  (axes[1], T, "Exact", "RdBu_r"),
                                  (axes[2], E, "|Pred - Exact|", "hot")):
        im = ax.pcolormesh(Xg, Tg, data, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Scale-PINN solution: {name}")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{base_name}_solution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")
    return float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true))


#时间切片图（五个时间点上的预测与参考解对比）
def draw_slice(output_dir, base_name, evaluator, name):
    slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4), sharey=True)
    for ax, t in zip(axes, slices):
        x, u_pred, u_true = evaluator.predict_slice(t)
        ax.plot(x, u_pred, "r-", label="pred")
        ax.plot(x, u_true, "k--", label="exact")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("u")
    fig.suptitle(f"Scale-PINN solution slices: {name}")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{base_name}_solution_slice.png")
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
    #运行标记：与 main_scale.py 保持一致（只改读取路径，不建目录）
    if args.tag:
        output_dir = os.path.join(output_dir, args.tag)
        base_name = f"{base_name}_{args.tag}"
    print(f"example name: {args.name}")
    print(f"version: {args.version}")
    print(f"device: {torch.get_default_device()}")
    example = create_example(args.name, DATASET_DIR, merged)

    print("=" * 60)
    final_l2 = draw_history(output_dir, base_name, os.path.join(output_dir, "csv"),
                            args.name, args.ipinn_dir)
    for label, value in final_l2.items():
        print(f"final RL2 [{label}] = {value:.6e}")
    print("=" * 60)

    evaluator = ScaleEvaluator(merged, example, base_name, output_dir)
    evaluator.load_model()
    grid_l2 = draw_solution(output_dir, base_name, evaluator, args.name)
    print(f"RL2 on reference grid ({args.name}) = {grid_l2:.6e}")
    draw_slice(output_dir, base_name, evaluator, args.name)
    print("done")
