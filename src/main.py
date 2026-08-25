import argparse
import os
from examples import create_example
from utils import load_yaml, path_init
from trainers import trainer_factory



def get_args():
    parser = argparse.ArgumentParser(description="run PINN Bench")
    parser.add_argument("-n", "--name", default="Helmholtz2D", help="name of example")
    parser.add_argument("-v", "--version", default="standard", help="name of version")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path, output_dir, DATASET_DIR, base_name = path_init(BASE_DIR, args.name, args.version)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_yaml(config_path)
    example = create_example(args.name, DATASET_DIR, config)
    trainer_factory.launch_trainer(config, example, output_dir)
