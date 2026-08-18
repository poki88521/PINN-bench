from . import StandardTrainer, ImprovedTrainer


def launch_trainer(config, example, example_dir):
    trainer_name = config.training.trainer_name
    if trainer_name == "StandardTrainer":
        StandardTrainer.launch(config, example, example_dir)
    elif trainer_name == "ImprovedTrainer":
        ImprovedTrainer.launch(config, example, example_dir)
    else:
        raise ValueError("Unknown trainer")