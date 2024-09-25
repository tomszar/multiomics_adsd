from momics_ad.figures import plots
from momics_ad.io import read
from momics_ad.stats import sd


def main():
    """
    Figure generation main routine.
    """
    # Read necessary data
    x_scores = read.read_xscores()
    x_center = sd.center_matrix(x_scores)
    metabolomics = read.read_metabolomics()
    metabolomics = metabolomics.drop(columns="DX")

    # Scatter plots
    plots.scatter_plot(x_center)
    # Line plots
    plots.orientation_plot(x_center)
    # Correlation plot
    plots.cor_plot(metabolomics)
