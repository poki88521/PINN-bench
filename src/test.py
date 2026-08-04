import os
from examples import AllenCahn1D


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
    _, _, dataset_dir = path_init('AllenCahn1D', "standard")
    example = AllenCahn1D(dataset_dir=dataset_dir)
    example.load_data()