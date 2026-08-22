import time
import os
from models import create_model
from trainers.monitors import TrainMonitor
from utils import CSVUtils


class StandardTrainer:
    def __init__(self, config, example, output_dir):
        self.config = config
        self.example = example
        self.output_dir = output_dir
        self.data = self.create_dataset(self.config.data)
        self.model = create_model(config, self.data)

    # 路径初始化
    def path_init(self):
        model_path = os.path.join(self.output_dir,
                                  f"{self.config.example.name}_{self.config.version}_model")
        csv_path = os.path.join(self.output_dir,
                                f"{self.config.example.name}_{self.config.version}_name.csv")
        return model_path, csv_path

    # 主训练函数
    def train(self, training_config, model_path, monitor):
        self.model.compile(training_config.optimizer, lr=training_config.lr)
        time_start = time.time()
        loss_history, train_state = self.model.train(iterations=training_config.iterations,
                                                batch_size=training_config.batch_size,
                                                model_save_path=model_path,
                                                display_every=training_config.display_every,
                                                callbacks=[monitor])
        time_elapsed = time.time() - time_start
        print(f"elapsed time: {time_elapsed:.2f}s")
        return loss_history, train_state, self.model

    # 创建数据集
    def create_dataset(self, data_config):
        if data_config.num_initial == 0:
            data = self.example.get_data(data_config.num_domain, data_config.num_boundary,
                                         data_config.num_test)
        else:
            data = self.example.get_data(data_config.num_domain, data_config.num_boundary,
                                    data_config.num_test, data_config.num_initial)
        return data

    # 保存csv文件
    def save(self, loss_history, train_state, history_path, training_config, monitor):
        CSVUtils.history_writer(history_path, loss_history, training_config)
        CSVUtils.info_writer(history_path, train_state)
        CSVUtils.component_writer(history_path, loss_history, self.example)
        CSVUtils.l2_error_writer(history_path, monitor)
        #?
        if hasattr(monitor, "sigma_names"):
            CSVUtils.sigma_writer(history_path, monitor)
        pass

    # 主启动函数
    def launch(self, config, example, output_dir):
        model_path, csv_path = self.path_init()
        monitor = TrainMonitor(self.example, config)
        loss_history, train_state, model = self.train(config.training, model_path, monitor)
        self.save(loss_history, train_state, csv_path, config.training, monitor)


