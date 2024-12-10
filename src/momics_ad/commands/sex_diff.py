import argparse

import pandas as pd

from momics_ad.io import read, subset
from momics_ad.stats import sd


def main():
    """
    Sex difference analysis main routine.
    """
    parser = argparse.ArgumentParser(description="Sex difference analysis.")
    parser.add_argument(
        "-I",
        type=int,
        default=999,
        metavar="ITERATIONS",
        help="Number of iterations to run the randomization.\
                        Default 999.",
    )
    parser.add_argument(
        "-F",
        type=str,
        default="pls",
        metavar="FILE",
        help="Type of file to use to generate the sex difference analysis.",
    )
    args = parser.parse_args()
    if args.F == "pls":
        x_scores = read.read_xscores()
    elif args.F == "snf":
        x_scores = read.read_spectral()
    else:
        Warning("No proper file name to read")
    x_center = sd.center_matrix(x_scores)
    X, Y = subset.get_XY(x_center)
    model_full = sd.get_model_matrix(X)
    model_red = sd.get_model_matrix(X, full=False)
    contrast = [[0, 1, 2], [3, 4, 5]]
    # Estimate LS vectors
    x_ls_full = sd._get_ls_vectors()
    delta, angle, shape = sd.estimate_difference(Y, model_full, x_ls_full, contrast)
    deltas, angles, shapes = sd.RRPP(
        Y, model_full, model_red, x_ls_full, contrast, args.I
    )
    total_rep = args.I + 1
    pvals = [
        (sum(angles > angle) / total_rep)[0, 1],
        (sum(deltas > delta) / total_rep)[0, 1],
        (sum(shapes > shape) / total_rep)[0, 1],
    ]
    vals = {
        "Values": [delta[0, 1], angle[0, 1], shape[0, 1]],
        "Index": ["delta", "angle", "shape"],
        "Pvalues": pvals,
    }
    result_table = pd.DataFrame(vals).set_index("Index")
    result_table.to_csv("ResultTable.csv")
