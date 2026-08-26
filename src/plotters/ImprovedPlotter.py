import matplotlib.pyplot as plt
import os
import numpy as np
from plotters import StandardPlotter
from utils import LoaderObject, ImprovedLoader


class ImprovedPlotter(StandardPlotter):
    def __init__(self, csv_dir, config, base_name, evaluator, only=None):
        # 对照组 loader：指向同级 control 目录（与 ImprovedTrainer.after_train 的约定一致）
        control_dir = os.path.join(csv_dir, "control")
        control_base = f"{config.example.name}_control"
        self.loader_c = LoaderObject(control_dir, control_base)
        super().__init__(csv_dir, config, base_name, evaluator, only)

    def get_loader(self):
        return ImprovedLoader(self.csv_dir, self.base_name)

    #sigma变化图
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

    #历史损失曲线（含对照组）
    def history_compare_plot(self):
        print("drawing history compare plot (ipinn vs control)...")
        iteration, train_loss, test_loss = self.loader.history()
        try:
            iteration_c, train_loss_c, test_loss_c = self.loader_c.history()
        except FileNotFoundError as e:
            raise RuntimeError(
                "Control-group data doesn't exist. Run ipinn train first."
                f"\ninexistent file: {e}") from e
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

    #损失分量图（含对照组）
    def components_compare_plot(self):
        print("drawing components compare plot (ipinn vs control)...")
        steps, train_dict, test_dict, names = self.loader.components()
        try:
            steps_c, train_dict_c, test_dict_c, names_c = self.loader_c.components()
        except FileNotFoundError as e:
            raise RuntimeError(
                "Control-group data doesn't exist. Run ipinn train first."
                f"\ninexistent file: {e}") from e
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
        fig.suptitle("Components Compare")
        axes = np.atleast_1d(axes)
        for (ax, name) in zip(axes, names):
            ax.semilogy(steps, np.clip(train_dict[name], 1e-12, None),
                        label=f"{self.config.version} train")
            ax.semilogy(steps, np.clip(test_dict[name], 1e-12, None),
                        label=f"{self.config.version} test")
            if name in train_dict_c:
                ax.semilogy(steps_c, np.clip(train_dict_c[name], 1e-12, None),
                            label="control train")
                ax.semilogy(steps_c, np.clip(test_dict_c[name], 1e-12, None),
                            label="control test")
            ax.set_title(name)
            ax.legend()
        fig.tight_layout()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_components_compare.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")

    #l2误差图（含对照组）
    def l2_compare_plot(self):
        print("drawing l2 compare plot (ipinn vs control)...")
        step, l2 = self.loader.l2()
        try:
            step_c, l2_c = self.loader_c.l2()
        except FileNotFoundError as e:
            raise RuntimeError(
                "Control-group data doesn't exist. Run ipinn train first."
                f"\ninexistent file: {e}") from e
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("L2 Compare")
        ax.plot(step, l2, label=f"{self.config.version} L2")
        ax.plot(step_c, l2_c, label="control L2")
        ax.set_xlabel("step")
        ax.set_ylabel("L2 error")
        ax.legend()
        out_path = os.path.join(self.csv_dir,
                                f"{self.config.example.name}_{self.config.version}_l2_compare.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")
