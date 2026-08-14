import os
from examples import *
from utils import load_yaml
from trainers import StandardTrainer

classes = {"Heat1D": Heat1D,
           "Helmholtz2D": Helmholtz2D,
           "Wave1D": Wave1D,
           "AllenCahn1D": AllenCahn1D,
           "Burgers1D": Burgers1D}

def path_init(name, version):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    dataset_dir = os.path.join(BASE_DIR, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    RUNS_DIR = os.path.join(BASE_DIR, 'runs')
    os.makedirs(RUNS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f'{name}_{version}.yaml')
    example_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(example_dir, exist_ok=True)
    return config_path, example_dir, dataset_dir

if __name__ == '__main__':
    name = "Helmholtz2D"
    version = "standard"

    config_path, example_dir, dataset_dir = path_init(name, version)
    config = load_yaml(config_path)
    example = classes[name](dataset_dir, config.example)
    StandardTrainer.launch(config, example, example_dir)
