import numpy as np
from examples import Wave1D
from trainers import StandardTrainer


def test(example, model, data_config):
    x_test = example.geomtime.uniform_points(n=data_config.num_test, boundary=True)
    u_true = example.u_exact_numpy(x_test)
    u_pred = model.predict(x_test)
    error = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {error:.2e}")
    pass

def launch(config, example_dir, dataset_dir):
    example = Wave1D(config.example.c)
    StandardTrainer.launch(config, example, example_dir, test)