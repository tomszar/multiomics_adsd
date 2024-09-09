from momics_ad.figures import plots
from momics_ad.io import read
from momics_ad.stats import sd


def main():
    """
    Figure generation main routine.
    """
    x_scores = read.read_xscores()
    x_center = sd.center_matrix(x_scores)
    plots.scatter_plot(x_center)
