from plotters import *

PLOTTERS = {"standard": StandardPlotter,
            "ipinn": ImprovedPlotter}

def draw_plots(version_name, only, csv_dir, config, base_name, evaluator):
    PLOTTERS[version_name](csv_dir, config, base_name, evaluator, only)