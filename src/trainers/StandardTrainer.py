from trainers import TrainerObject

#std版本的训练器（模板功能已经被独立出去了）
class StandardTrainer(TrainerObject):
    def __init__(self, config, example, output_dir, base_name, base_config=None):
        super().__init__(config, example, output_dir, base_name, base_config)




