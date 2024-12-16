import argparse

import pandas as pd

from momics_ad.figures import plots
from momics_ad.io import read
from momics_ad.stats import sd


def main():
    """
    Figure generation main routine.
    """
    parser = argparse.ArgumentParser(description="Generate figures.")
    parser.add_argument(
        "-F",
        type=str,
        default="pls",
        metavar="FILE",
        help="Type of file to use to generate the figures.",
    )
    # Read necessary data
    args = parser.parse_args()
    x_scores = pd.DataFrame([])
    vips = pd.DataFrame([])
    if args.F == "pls":
        x_scores = read.read_xscores()
        vips = pd.read_csv("VIPS.csv", header=None)
    elif args.F == "snf":
        x_scores = read.read_spectral()
    else:
        Warning("No proper file name to read.")
    x_center = sd.center_matrix(x_scores)
    metabolomics = read.read_metabolomics()
    met_names = []
    concat_met = metabolomics["p180"].merge(metabolomics["nmr"], on="RID")
    for key in metabolomics:
        if key != "qt":
            col_names = list(metabolomics[key].columns)
            for col in col_names:
                met_names.append(col)

    # Line plots
    plots.orientation_plot(x_center)
    # Correlation plot
    plots.cor_plot(concat_met)

    if args.F == "pls":
        plots.vip_plot(vips, met_names)
        plots.scatter_plot(
            x_center,
            xlim=(-4, 4),
            ylim=(-4, 4),
            filename="ScatterPlot_PLS.pdf",
        )
    elif args.F == "snf":
        plots.scatter_plot(
            x_center,
            xlim=(-0.02, 0.02),
            ylim=(-0.02, 0.02),
            filename="ScatterPlot_SNF.pdf",
        )
