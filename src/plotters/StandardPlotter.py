import os
import numpy as np
import matplotlib.pyplot as plt
from utils import LoaderObject, Evaluator

#std版本的绘图类，同时是所有绘图类的父类
#此类中的图是各种版本必须绘制的图
class StandardPlotter:
    def __init__(self, csv_dir, config, base_name, evaluator, only=None):
        self.csv_dir = csv_dir
        self.config = config
        self.base_name = base_name
        #重运行模型的评估类实例
        self.evaluator = evaluator
        #数据加载类
        self.loader = self.get_loader()
        #遍历执行获取到的画图方法
        for plot_func in self.resolve_plots(only):
            plot_func()

    #获取数据加载器，可重写
    def get_loader(self):
        return LoaderObject(self.csv_dir, self.base_name)

    # 当前类可用的全部绘图方法名（以_plot 结尾）
    def plot_methods(self):
        return sorted(m for m in dir(self) if m.endswith("_plot"))

    # 把 only 图名列表解析成绘图函数列表（没有only就返回所有）
    def resolve_plots(self, only):
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

    #训练时的loss历史（含test和train）
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

    #损失分量图
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

    #l2 error图
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

    #预测场热力图（预测 | 精确 | 绝对误差）
    def solution_plot(self):
        print("drawing solution plot...")
        x_test, u_pred, u_true, abs_err = self.evaluator.predict()
        nx = len(np.unique(x_test[:, 0]))
        ny = len(np.unique(x_test[:, 1]))
        X = x_test[:, 0].reshape(ny, nx)
        Y = x_test[:, 1].reshape(ny, nx)
        U = u_pred.reshape(ny, nx)
        T = u_true.reshape(ny, nx)
        E = abs_err.reshape(ny, nx)
        if self.evaluator.example.time_domain is not None:
            xlabel, ylabel = "x", "t"
        else:
            xlabel, ylabel = "x", "y"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, data, title, cmap in [
                (axes[0], U, "Predicted", "RdBu_r"),
                (axes[1], T, "Exact", "RdBu_r"),
                (axes[2], E, "|Pred - Exact|", "hot")]:
            im = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"Solution: {self.config.example.name}")
        fig.tight_layout()
        out_path = os.path.join(self.csv_dir, f"{self.base_name}_solution.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")

    #时间切片图（取时间点为0, 0.25, 0.5, 0.75, 1五个时间点）
    def solution_slice_plot(self):
        print("drawing solution slice plot...")
        if self.evaluator.example.time_domain is None:
            print("skip: not a time-dependent example")
            return
        time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
        n = len(time_slices)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
        for (ax, t) in zip(axes, time_slices):
            x, u_pred, u_true = self.evaluator.predict_slice(t)
            ax.plot(x, u_pred, "r-", label="pred")
            ax.plot(x, u_true, "k--", label="exact")
            ax.set_title(f"t = {t:.2f}")
            ax.set_xlabel("x")
            ax.grid(alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("u")
        fig.suptitle(f"Solution slices: {self.config.example.name}")
        fig.tight_layout()
        out_path = os.path.join(self.csv_dir, f"{self.base_name}_solution_slice.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved: {out_path}")