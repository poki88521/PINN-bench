import os

def path_init(base_dir, name, version):
    configs_dir = os.path.join(base_dir, 'configs', version)
    config_path = os.path.join(configs_dir, f'{name}_{version}.yaml')
    dataset_dir = os.path.join(base_dir, 'dataset')
    example_dir = os.path.join(base_dir, 'runs', name)
    return config_path, example_dir, dataset_dir