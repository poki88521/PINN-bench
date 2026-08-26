import argparse
import os
import plotters
from examples import create_example
from utils import load_yaml, path_init, get_version_config, Evaluator


def get_args():
    parser = argparse.ArgumentParser(description="draw PINN bench plots")
    parser.add_argument("-n", "--name", default="Helmholtz2D", help="name of example")
    parser.add_argument("-v", "--version", default="standard", help="name of version")
    parser.add_argument("-o", "--only", nargs="*", default=None,
                        help="only draw plots that name is given, or draw all plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path, csv_dir, DATASET_DIR, base_name = path_init(BASE_DIR, args.name, args.version)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_yaml(config_path)
    merged = get_version_config(config)
    example = create_example(args.name, DATASET_DIR, merged)
    evaluator = Evaluator(merged, example, base_name, csv_dir)
    plotters.draw_plots(args.version, args.only, csv_dir, merged, base_name, evaluator)
