from examples import *


CLASSES = {"Heat1D": Heat1D,
           "Helmholtz2D": Helmholtz2D,
           "Wave1D": Wave1D,
           "AllenCahn1D": AllenCahn1D,
           "Burgers1D": Burgers1D}


def create_example(name, dataset_dir, config):
    #校验算例名称
    if name not in CLASSES:
        raise ValueError(f"Unknown example name '{name}'. Available: {list(CLASSES)}")
    return CLASSES[name](dataset_dir, config.example)