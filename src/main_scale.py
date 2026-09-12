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
    return parser.parse_args()


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
