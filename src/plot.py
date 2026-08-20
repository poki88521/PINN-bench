import argparse
import os
import plotters
from utils import load_yaml, path_init


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
    config_path, RUNS_DIR, DATASET_DIR = path_init(BASE_DIR, args.name, args.version)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_yaml(config_path)
    csv_dir = os.path.join(RUNS_DIR, config.version)
    plotters.draw_plots(args.version, args.only, csv_dir, config)
