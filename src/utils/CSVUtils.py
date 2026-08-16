import numpy as np
import csv

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


def component_writer(history_path, loss_history, example):
    component_path = history_path.replace("_history.csv", "_components.csv")
    n_components = len(loss_history.loss_train[0])
    names = COMPONENT_NAMES.get(example.name)
    if names is None or len(names) != n_components:
        names = (["PDE"]
                 + [f"BC{i}({type(bc).__name__})" for i, bc in enumerate(example.bcs, 1)]
                 + [f"IC{i}({type(ic).__name__})" for i, ic in enumerate(example.ics, 1)])
    with open(component_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "component", "loss"])
        for i, (tr, te) in enumerate(zip(loss_history.loss_train,
                                         loss_history.loss_test)):
            step = loss_history.steps[i] if i < len(loss_history.steps) else i
            for name, value in zip(names, tr):
                writer.writerow([step, "train", name, f"{value:.6e}"])
            for name, value in zip(names, te):
                writer.writerow([step, "test", name, f"{value:.6e}"])


def history_writer(history_path, loss_history, training_config):
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'train_loss(sum)', 'test_loss(sum)'])
        #print(f"total num of loss:{len(loss_history.loss_train)}")
        for i, (tr, te) in enumerate(zip(loss_history.loss_train, loss_history.loss_test)):
            writer.writerow([i * training_config.display_every, np.sum(tr), np.sum(te)])
    pass

def info_writer(info_path, train_state):
    with open(info_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_step', "best_loss_train", "best_loss_test"])
        writer.writerow([train_state.best_step, train_state.best_loss_train, train_state.best_loss_test])
    pass