import os
from examples import *
from utils import load_yaml, path_init
from trainers import trainer_factory

CLASSES = {"Heat1D": Heat1D,
           "Helmholtz2D": Helmholtz2D,
           "Wave1D": Wave1D,
           "AllenCahn1D": AllenCahn1D,
           "Burgers1D": Burgers1D}


if __name__ == '__main__':
    name = "AllenCahn1D"
    version = "ipinn"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path, example_dir, dataset_dir = path_init(BASE_DIR, name, version)
    config = load_yaml(config_path)
    example = CLASSES[name](dataset_dir, config.example)
    trainer_factory.launch_trainer(config, example, example_dir)
