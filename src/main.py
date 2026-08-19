import argparse
import os
from examples import *
from utils import load_yaml, path_init
from trainers import trainer_factory

CLASSES = {"Heat1D": Heat1D,
           "Helmholtz2D": Helmholtz2D,
           "Wave1D": Wave1D,
           "AllenCahn1D": AllenCahn1D,
           "Burgers1D": Burgers1D}

def get_args():
    parser = argparse.ArgumentParser(description="run PINN Bench")
    parser.add_argument("-n", "--name", default="Helmholtz2D", help="name of example")
    parser.add_argument("-v", "--version", default="standard", help="name of version")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.name not in CLASSES:
        raise ValueError(f"Unknown example name '{args.name}'. Available: {list(CLASSES)}")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path, example_dir, dataset_dir = path_init(BASE_DIR, args.name, args.version)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_yaml(config_path)
    example = CLASSES[args.name](dataset_dir, config.example)
    trainer_factory.launch_trainer(config, example, example_dir)
