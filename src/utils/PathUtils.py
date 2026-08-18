import os

def path_init(base_dir, name, version):
    configs_dir = os.path.join(base_dir, 'configs', version)
    os.makedirs(configs_dir, exist_ok=True)
    config_path = os.path.join(configs_dir, f'{name}_{version}.yaml')

    dataset_dir = os.path.join(base_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    example_dir = os.path.join(base_dir, 'runs', name)
    os.makedirs(example_dir, exist_ok=True)

    return config_path, example_dir, dataset_dir