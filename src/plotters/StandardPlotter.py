import os
import numpy as np
import matplotlib.pyplot as plt
from utils import LoaderObject


class StandardPlotter:
    def __init__(self, csv_dir, config, base_name, only=None):
        self.csv_dir = csv_dir
        self.config = config
        self.base_name = base_name
        self.loader = self.get_loader()
        for plot_func in self.resolve_plots(only):
            plot_func()

    def get_loader(self):
        return LoaderObject(self.csv_dir, self.base_name)

    def plot_methods(self):
        #当前类可用的全部绘图方法名（以_plot 结尾）
        return sorted(m for m in dir(self) if m.endswith("_plot"))

    def resolve_plots(self, only):
        #把 only 图名列表解析成绘图函数列表
        if only is None:
            #返回所有的绘图（_plot结尾）方法
            return [getattr(self, m) for m in self.plot_methods()]
        funcs = []
        for name in only:
            fn = getattr(self, name, None)
            if fn is None or not callable(fn):
                raise ValueError(
                    f"Unknown plot '{name}' in {type(self).__name__}. "
                    f"Available: {self.plot_methods()}")
            funcs.append(fn)
        return funcs

    def history_plot(self):
        print("drawing total loss plot...")
        iteration, train_loss, test_loss = self.loader.history()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Total Loss")
        ax.plot(iteration, train_loss, label="Train")
        ax.plot(iteration, test_loss, label="Test")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_history.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")
        pass

    def components_plot(self):
        print("drawing components loss plot...")
        steps, train_dict, test_dict, names = self.loader.components()
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
        fig.suptitle("Components Loss")
        axes = np.atleast_1d(axes)
        for (ax, name) in zip(axes, names):
            y_train = np.clip(train_dict[name], 1e-12, None)
            y_test = np.clip(test_dict[name], 1e-12, None)
            ax.semilogy(steps, y_train, label="train")
            ax.semilogy(steps, y_test, label="test")
            ax.set_title(name)
            ax.legend()
        fig.tight_layout()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_components.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")

    def l2_error_plot(self):
        print("drawing l2 error plot...")
        step, l2 = self.loader.l2()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("L2 Error")
        ax.plot(step, l2, label="L2 error")
        ax.set_xlabel("step")
        ax.set_ylabel("L2 error")
        ax.legend()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_l2.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")
        pass


