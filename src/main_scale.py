import argparse
import os

import torch

from examples import create_example
from trainers import ScaleTrainer
from utils import load_yaml, path_init, get_version_config


def get_args():
    parser = argparse.ArgumentParser(description="run Scale-PINN")
    parser.add_argument("-n", "--name", default="AllenCahn1D", help="name of example")
    parser.add_argument("-v", "--version", default="scale", help="name of version")
    parser.add_argument("--iterations", type=int, default=None,
                        help="覆盖配置里的迭代数（冒烟测试用，默认读配置）")
    parser.add_argument("--display_every", type=int, default=None,
                        help="覆盖配置里的记录间隔（冒烟测试用，默认读配置）")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖配置里的随机种子（默认读配置）")
    parser.add_argument("--ER", type=float, default=None,
                        help="覆盖配置里的 ER；置 0 即官方 ER=0 对照组（关闭序贯修正）")
    parser.add_argument("--tag", default=None,
                        help="运行标记：产物落到 runs/<name>/<version>/<tag>/，"
                             "文件名加 _<tag> 后缀（避免 ER=0、多种子运行互相覆盖）")
    return parser.parse_args()


#运行标记：把本次运行的产物隔离到子目录，并同步修改文件名前缀
def apply_tag(output_dir, base_name, tag):
    if not tag:
        return output_dir, base_name
    output_dir = os.path.join(output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, f"{base_name}_{tag}"


if __name__ == '__main__':
    args = get_args()
    #设备：有 GPU 用 cuda，否则 cpu（与 main.py 一致）
    torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path, output_dir, DATASET_DIR, base_name = path_init(BASE_DIR, args.name, args.version)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_yaml(config_path)
    #合并同名覆盖节后再取参数（trainer_name/lr/iterations 等以 scale 节为准）
    merged = get_version_config(config)
    if args.iterations is not None:
        merged.training.iterations = args.iterations
    if args.display_every is not None:
        merged.training.display_every = args.display_every
    if args.seed is not None:
        merged.scale.seed = args.seed
    if args.ER is not None:
        merged.scale.ER = args.ER
    #运行标记：隔离本次运行的产物目录与文件名
    output_dir, base_name = apply_tag(output_dir, base_name, args.tag)
    print(f"example name: {args.name}")
    print(f"version: {args.version}")
    print(f"device: {torch.get_default_device()}")
    print(f"iterations: {merged.training.iterations}, "
          f"display_every: {merged.training.display_every}, lr: {merged.training.lr}")
    print(f"ER: {merged.scale.ER}, ER_xx: {merged.scale.ER_xx}, "
          f"weight_ic: {merged.scale.weight_ic}, weight_bc: {merged.scale.weight_bc}")
    print(f"batch: domain={merged.scale.batch_domain}, "
          f"initial={merged.scale.batch_initial}, boundary={merged.scale.batch_boundary}")
    print(f"seed: {merged.scale.seed}")
    print(f"output dir: {output_dir}")
    example = create_example(args.name, DATASET_DIR, merged)
    trainer = ScaleTrainer(merged, example, output_dir, base_name, base_config=config)
    trainer.launch()
