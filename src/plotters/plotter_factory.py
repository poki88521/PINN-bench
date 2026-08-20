from plotters import *

PLOTTERS = {"standard": StandardPlotter}

def draw_plots(version_name, only, csv_dir, config):
    PLOTTERS[version_name](csv_dir, config, only)