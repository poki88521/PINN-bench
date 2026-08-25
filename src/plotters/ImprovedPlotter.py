import matplotlib.pyplot as plt
import os
import numpy as np
from plotters import StandardPlotter
from utils import LoaderObject, ImprovedLoader


class ImprovedPlotter(StandardPlotter):
    def __init__(self, csv_dir, config, base_name, only=None):
        # 对照组 loader：指向同级 control 目录（与 ImprovedTrainer.after_train 的约定一致）
        control_dir = os.path.join(os.path.dirname(csv_dir), "control")
        control_base = f"{config.example.name}_control"
        self.loader_c = LoaderObject(control_dir, control_base)
        super().__init__(csv_dir, config, base_name, only)

    def get_loader(self):
        return ImprovedLoader(self.csv_dir, self.base_name)

    def get_plot(self):
        return

    def sigma_plot(self):
        print("drawing sigma plot...")
        steps, sigma_dict, names = self.loader.sigma()
        fig, axs = plt.subplots(1, len(sigma_dict), figsize=(5 * len(sigma_dict), 5))
        fig.suptitle("Sigma plot")
        axs = np.atleast_1d(axs)
        for (name, ax) in zip(names, axs):
            ax.plot(steps, sigma_dict[name], label=name)
            ax.set_title(name)
            ax.set_xlabel("step")
            ax.set_ylabel(name)
            ax.legend()
        fig.tight_layout()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_sigma.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")

    def history_compare_plot(self):
        print("drawing history compare plot (ipinn vs control)...")
        iteration, train_loss, test_loss = self.loader.history()
        try:
            iteration_c, train_loss_c, test_loss_c = self.loader_c.history()
        except FileNotFoundError as e:
            raise RuntimeError(
                "对照组数据不存在，无法绘制对比图。请先运行 ipinn 训练"
                "（ImprovedTrainer.after_train 会自动生成 control 对照组）。"
                f"\n缺失文件: {e}") from e
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("History Compare")
        ax.plot(iteration, train_loss, label=f"{self.config.version} train")
        ax.plot(iteration, test_loss, label=f"{self.config.version} test")
        ax.plot(iteration_c, train_loss_c, label="control train")
        ax.plot(iteration_c, test_loss_c, label="control test")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_history_compare.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")