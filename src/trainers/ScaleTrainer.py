from trainers import TrainerObject
from models import ScaleModel as scale


class ScaleTrainer(TrainerObject):
    def __init__(self, config, example, output_dir, base_name, base_config=None):
        super().__init__(config, example, output_dir, base_name, base_config)
        dims_config = config.dims
        self.dims = [dims_config.in_dim] + [dims_config.width] * dims_config.depth + [dims_config.out_dim]

    def train(self, training_config):
        model = scale.PINN(self.dims, self.config.model)
        model_pre = scale.DNN(self.dims, self.config.model)
        model_pre.load_state_dict(model.state_dict())

        for it in range(self.config.training.iterations):
            
