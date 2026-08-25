from . import StandardTrainer, ImprovedTrainer
from utils import get_version_config

TRAINERS = {
    "StandardTrainer": StandardTrainer,
    "ImprovedTrainer": ImprovedTrainer,
}


def launch_trainer(config, example, output_dir):
    # 合并版本覆盖配置：顶层为 std 基线，同名节（如 ipinn:）覆盖后分派
    merged = get_version_config(config)
    trainer_name = merged.training.trainer_name
    if trainer_name not in TRAINERS:
        raise ValueError(f"Unknown trainer: {trainer_name}")
    base_name = f"{merged.example.name}_{merged.version}"
    trainer = TRAINERS[trainer_name](merged, example, output_dir, base_name,
                                     base_config=config)
    trainer.launch()
