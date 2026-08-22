import os.path
import numpy as np
import csv
import pandas as pd


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


def component_writer(csv_path, loss_history, example):
    component_path = csv_path.replace("_name.csv", "_components.csv")
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


def history_writer(csv_path, loss_history, training_config):
    history_path = csv_path.replace("_name.csv", "_history.csv")
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'train_loss(sum)', 'test_loss(sum)'])
        for i, (tr, te) in enumerate(zip(loss_history.loss_train, loss_history.loss_test)):
            writer.writerow([i * training_config.display_every, np.sum(tr), np.sum(te)])
    pass


def l2_error_writer(csv_path, monitor):
    info_path = csv_path.replace("_name.csv", "_l2.csv")
    with open(info_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', "l2_error"])
        for (step, l2) in zip(monitor.steps, monitor.l2_errors):
            writer.writerow([step, l2])
    pass


def sigma_writer(csv_path, monitor):
    sigma_path = csv_path.replace("_name.csv", "_sigma.csv")
    with open(sigma_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + monitor.sigma_names)
        for i, step in enumerate(monitor.sigma_steps):
            writer.writerow([step] + [monitor.sigma_values[name][i]
                                      for name in monitor.sigma_names])
    pass


def info_writer(csv_path, train_state):
    info_path = csv_path.replace("_name.csv", "_info.csv")
    with open(info_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_step', "best_loss_train", "best_loss_test"])
        writer.writerow([train_state.best_step, train_state.best_loss_train, train_state.best_loss_test])
    pass


def load_history(csv_dir, config):
    log = pd.read_csv(os.path.join(csv_dir, f"{config.example.name}_{config.version}_history.csv"))
    log = log.dropna(how="all")
    return log["iteration"], log["train_loss(sum)"], log["test_loss(sum)"]


def load_components(csv_dir, config):
    path = os.path.join(csv_dir, f"{config.example.name}_{config.version}_components.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Components CSV not found: {path}")
    log = pd.read_csv(path)
    log = log.dropna(how="all")
    train = log[log["split"] == "train"]
    test = log[log["split"] == "test"]
    component_names = list(dict.fromkeys(log["component"].tolist()))
    steps = train.loc[train["component"] == component_names[0], "step"].to_numpy()
    train_dict = {name: train.loc[train["component"] == name, "loss"].to_numpy()
                  for name in component_names}
    test_dict = {name: test.loc[test["component"] == name, "loss"].to_numpy()
                 for name in component_names}
    return steps, train_dict, test_dict, component_names


def load_l2(csv_dir, config):
    log = pd.read_csv(os.path.join(csv_dir, f"{config.example.name}_{config.version}_l2.csv"))
    log = log.dropna(how="all")
    return log["step"], log["l2_error"]


def load_sigma(csv_dir, config):
    path = os.path.join(csv_dir, f"{config.example.name}_{config.version}_sigma.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sigma CSV not found: {path}")
    log = pd.read_csv(path)
    log = log.dropna(how="all")
    steps = log["step"].to_numpy()
    names = [col for col in log.columns if col != "step"]
    sigma_dict = {col: log[col].to_numpy() for col in names}
    return steps, sigma_dict, names