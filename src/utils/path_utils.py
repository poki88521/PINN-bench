import os

def path_init(base_dir, name, version):
    configs_dir = os.path.join(base_dir, 'configs', version)
    config_path = os.path.join(configs_dir, f'{name}_{version}.yaml')
    dataset_dir = os.path.join(base_dir, 'dataset')
    runs_dir = os.path.join(base_dir, 'runs')
    output_dir = os.path.join(runs_dir, name, version)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{name}_{version}"
    return config_path, output_dir, dataset_dir, base_name


