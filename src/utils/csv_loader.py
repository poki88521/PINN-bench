import os
import pandas as pd

#所有数据加载器的父类，同时作为std版本的加载器
class LoaderObject:
    def __init__(self, csv_dir, base_name):
        self.csv_dir = csv_dir
        self.base_name = base_name

    def _path(self, suffix):
        return os.path.join(self.csv_dir, f"{self.base_name}_{suffix}.csv")

    def _read_csv(self, suffix, required=True):
        path = self._path(suffix)
        if not os.path.isfile(path):
            if required:
                raise FileNotFoundError(f"CSV not found: {path}")
            return None
        return pd.read_csv(path).dropna(how="all")

    def history(self):
        log = self._read_csv("history")
        return log["iteration"], log["train_loss(sum)"], log["test_loss(sum)"]

    def components(self):
        log = self._read_csv("components")
        train = log[log["split"] == "train"]
        test = log[log["split"] == "test"]
        component_names = list(dict.fromkeys(log["component"].tolist()))
        steps = train.loc[train["component"] == component_names[0], "step"].to_numpy()
        train_dict = {name: train.loc[train["component"] == name, "loss"].to_numpy()
                      for name in component_names}
        test_dict = {name: test.loc[test["component"] == name, "loss"].to_numpy()
                     for name in component_names}
        return steps, train_dict, test_dict, component_names

    def l2(self):
        log = self._read_csv("l2")
        return log["step"], log["l2_error"]

#ipinn版本的加载器
class ImprovedLoader(LoaderObject):
    def sigma(self, required=False):
        log = self._read_csv("sigma", required=required)
        if log is None:
            return None
        steps = log["step"].to_numpy()
        names = [col for col in log.columns if col != "step"]
        sigma_dict = {col: log[col].to_numpy() for col in names}
        return steps, sigma_dict, names
