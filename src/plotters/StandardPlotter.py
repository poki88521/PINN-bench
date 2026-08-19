from utils import CSVUtils
import os
import numpy as np
import matplotlib.pyplot as plt

def history_plot(csv_dir, config):
    iteration, train_loss, test_loss = CSVUtils.load_history(csv_dir, config)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Total Loss")
    ax.plot(iteration, train_loss, label="Train")
    ax.plot(iteration, test_loss, label="Test")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(csv_dir, f"{config.example.name}_{config.version}_history.png")
    pass


def component_plot(csv_dir, config):
    steps, train_dict, test_dict, names = CSVUtils.load_components(csv_dir, config)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    fig.suptitle("Components Loss")
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        y_train = np.clip(train_dict[name], 1e-12, None)
        y_test = np.clip(test_dict[name], 1e-12, None)
        ax.semilogy(steps, y_train, label="train")
        ax.semilogy(steps, y_test, label="test")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    out_path = os.path.join(csv_dir, f"{config.example.name}_{config.version}_components.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")