import os

from trainers import HelmholtzTrainer

if __name__ == '__main__':
    name = "Helmholtz"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')
    RUNS_DIR = os.path.join(BASE_DIR, 'runs')
    config_path = os.path.join(CONFIGS_DIR, f'{name}.yaml')

    HelmholtzTrainer.launch(config_path)
