import os
import csv
import numpy as np


COMPONENT_NAMES = {
    "Helmholtz2D": ["PDE", "BC(exact u on boundary)"],
    "Heat1D": ["PDE", "BC(u=0 at x=0,1)", "IC(u=sin(pi*x))"],
    "Wave1D": ["PDE", "BC(u=0 at x=0)", "BC(u=0 at x=1)",
               "IC(u=sin(pi*x))", "IC(u_t=0)"],
    "AllenCahn1D": ["PDE", "BC(u=-1 at x=-1)", "BC(u=-1 at x=1)",
                    "IC(u=x^2*cos(pi*x))"],
    "Burgers1D": ["PDE", "BC(u=0 at x=-1)", "BC(u=0 at x=1)",
                  "IC(u=-sin(pi*x))"],
}


class WriterObject:
    def __init__(self, output_dir, base_name, monitor):
        self.output_dir = output_dir
        self.base_name = base_name
        self.monitor = monitor

    # 路径获取
    def _path(self, suffix):
        return os.path.join(self.output_dir, f"{self.base_name}_{suffix}.csv")

    # 行写入
    def _write_rows(self, suffix, header, rows):
        with open(self._path(suffix), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    # 历史损失
    def history(self, loss_history, training_config):
        header = ["iteration", "train_loss(sum)", "test_loss(sum)"]
        rows = [[i * training_config.display_every, np.sum(tr), np.sum(te)]
                for i, (tr, te) in enumerate(zip(loss_history.loss_train,
                                                 loss_history.loss_test))]
        self._write_rows("history", header, rows)

    # l2 误差
    def l2_error(self):
        header = ['step', "l2_error"]
        rows = [[step, l2] for step, l2 in zip(self.monitor.steps,
                                               self.monitor.l2_errors)]
        self._write_rows("l2", header, rows)

    # 损失分量
    def components(self, loss_history, example):
        n_components = len(loss_history.loss_train[0])
        names = COMPONENT_NAMES.get(example.name)
        if names is None or len(names) != n_components:
            names = (["PDE"]
                     + [f"BC{i}({type(bc).__name__})" for i, bc in enumerate(example.bcs, 1)]
                     + [f"IC{i}({type(ic).__name__})" for i, ic in enumerate(example.ics, 1)])
        rows = []
        for i, (tr, te) in enumerate(zip(loss_history.loss_train,
                                         loss_history.loss_test)):
            step = loss_history.steps[i] if i < len(loss_history.steps) else i
            for name, value in zip(names, tr):
                rows.append([step, "train", name, f"{value:.6e}"])
            for name, value in zip(names, te):
                rows.append([step, "test", name, f"{value:.6e}"])
        self._write_rows("components", ["step", "split", "component", "loss"], rows)

    # 其他信息
    def info(self, train_state):
        header = ['best_step', "best_loss_train", "best_loss_test"]
        rows = [[train_state.best_step,
                 train_state.best_loss_train,
                 train_state.best_loss_test]]
        self._write_rows("info", header, rows)



class ImprovedWriter(WriterObject):
    def __init__(self, output_dir, base_name, monitor):
        super().__init__(output_dir, base_name, monitor)

    def sigma(self):
        header = ["step"] + self.monitor.sigma_names
        rows = [[step] + [self.monitor.sigma_values[name][i]
                          for name in self.monitor.sigma_names]
                for i, step in enumerate(self.monitor.sigma_steps)]
        self._write_rows("sigma", header, rows)