import csv
from examples import Helmholtz2D
from utils import load_yaml
from trainers import StandardTrainer



def save(loss_history, train_state, history_path, training_config, info_path):
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'train_loss', 'test_loss'])
        print(f"total num of loss:{len(loss_history.loss_train)}")
        for i, (tr, te) in enumerate(zip(loss_history.loss_train, loss_history.loss_test)):
            writer.writerow([(i + 1) * training_config.display_every, tr, te])

    with open(info_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_iteration', "best_loss"])
        writer.writerow([train_state.best_iteration, train_state.best_loss])
    pass


def launch(path, example_dir):
    config = load_yaml(path)
    example = Helmholtz2D(a1=config.example.a1, a2=config.example.a2, k=config.example.k)
    version_dir, model_path, history_path, info_path = StandardTrainer.path_init(config, example_dir)
    data = StandardTrainer.create_dataset(example, config.data)
    model = StandardTrainer.create_model(data, config.dims)
    loss_history, train_state, model = StandardTrainer.train(model, config.training, model_path)
    StandardTrainer.test(example, model, config.data)
    save(loss_history, train_state, history_path, config.training, info_path)
