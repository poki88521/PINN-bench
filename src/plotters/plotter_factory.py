from plotters import *

#绘图类的字典，需要持续维护
PLOTTERS = {"standard": StandardPlotter,
            "ipinn": ImprovedPlotter}

#绘图器调度
def draw_plots(version_name, only, csv_dir, config, base_name, evaluator):
    PLOTTERS[version_name](csv_dir, config, base_name, evaluator, only)