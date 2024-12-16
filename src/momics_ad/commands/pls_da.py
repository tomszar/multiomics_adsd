import argparse

import pandas as pd

from momics_ad.io import read
from momics_ad.stats import pls


def main():
    """
    PLSR-DA main routine.
    """
    parser = argparse.ArgumentParser(
        description="Partial Least Squares Regression -\
        Discriminant Analysis."
    )
    parser.add_argument(
        "-R",
        type=int,
        default=30,
        metavar="REPEATS",
        help="Number of times the CV loop should be repeated.\
                        Default 30.",
    )
    parser.add_argument(
        "-C",
        type=int,
        default=50,
        metavar="COMPONENTS",
        help="Maximum number of components to evaluate.\
                        Default 50.",
    )
    args = parser.parse_args()
    m = read.read_metabolomics()
    X = m["nmr"].merge(m["p180"], on="RID")
    Y = m["qt"]["DX"]
    model_table = pls.plsda_doubleCV(
        X,
        Y,
        n_repeats=args.R,
        max_components=args.C,
    )
    i_b_m = model_table["table"].iloc[:, 2].idxmax()
    best_mod = model_table["models"][i_b_m]
    xscores = pd.DataFrame(best_mod.transform(X))
    xscores.index = X.index
    xscores.to_csv("Xscores.csv")
    vips = pls._calculate_vips(best_mod)
    pd.DataFrame(vips).to_csv("VIPS.csv", header=False, index=False)
    model_table["table"].to_csv("ModelTable.csv")
