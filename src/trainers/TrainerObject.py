import time
import os
from models import create_model
from trainers.monitors import TrainMonitor
from utils import CSVWriter


class TrainerObject:
    #所有trainer类的基类
    #使用方法:Trainer(...).launch()
    #init(dataset -> model -> preprocess) -> launch(monitor -> writer -> train -> save -> after_train)

    def __init__(self, config, example, output_dir, base_name, base_config=None):
        self.config = config
        self.base_config = base_config if base_config is not None else config
        self.example = example
        self.output_dir = output_dir
        self.base_name = base_name
        self.monitor = None
        self.writer = None
        self.data = self.create_dataset(self.config.data)
        self.model = create_model(self.config, self.data)
        self.preprocess()

    #模型保存路径（train 内部使用）
    @property
    def model_path(self):
        return os.path.join(self.output_dir, f"{self.base_name}_model")

    #启动函数，禁止重写
    def launch(self):
          self.monitor = self.make_monitor()
          self.writer = self.make_writer()
          loss_history, train_state = self.train(self.config.training)
          self.save(loss_history, train_state, self.config.training)
          self.after_train(loss_history, train_state, self.model)

    #预处理钩子，可重写
    def preprocess(self):
        pass

    #获取检测器钩子，可重写
    def make_monitor(self):
        return TrainMonitor(self.example, self.config)

    #获取写入器钩子，可重写
    def make_writer(self):
        return CSVWriter(self.output_dir, self.base_name, self.monitor)

    #训练后处理钩子，可重写
    def after_train(self, loss_history, train_state, model):
        pass

    #创建数据集
    def create_dataset(self, data_config):
          if data_config.num_initial == 0:
              return self.example.get_data(data_config.num_domain,
                                           data_config.num_boundary,
                                           data_config.num_test)
          return self.example.get_data(data_config.num_domain,
                                       data_config.num_boundary,
                                       data_config.num_test,
                                       data_config.num_initial)

    #主训练函数train(training_config) -> loss_history, train_state
    #由于launch()不可修改，train函数的参数和返回类型也不可修改
    def train(self, training_config):
        self.model.compile(training_config.optimizer, lr=training_config.lr)
        time_start = time.time()
        loss_history, train_state = self.model.train(iterations=training_config.iterations,
                                                     batch_size=training_config.batch_size,
                                                     model_save_path=self.model_path,
                                                     display_every=training_config.display_every,
                                                     callbacks=[self.monitor])
        time_elapsed = time.time() - time_start
        print(f"elapsed time: {time_elapsed:.2f}s")
        return loss_history, train_state

    def save(self, loss_history, train_state, training_config):
        self.writer.history(loss_history, self.config.training)
        self.writer.components(loss_history, self.example)
        self.writer.info(train_state)
        self.writer.l2_error()


