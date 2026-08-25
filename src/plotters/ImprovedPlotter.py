import matplotlib.pyplot as plt
import os
import numpy as np
from plotters import StandardPlotter


class ImprovedPlotter(StandardPlotter):
    def __init__(self, csv_dir, config, base_name, only=None):
        super().__init__(csv_dir, config, base_name, only)

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