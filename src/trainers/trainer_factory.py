
from trainers import StandardTrainer


def launch_trainer(config, example, example_dir):
    trainer_name = config.trainer_name
    if trainer_name == "StandardTrainer":
        StandardTrainer.launch(config, example, example_dir)
    else:
        pass