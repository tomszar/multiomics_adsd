import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from momics_ad.figures import plots
from momics_ad.stats import snf


def main():
    """
    Main SNF routine.
    """
    # Read metabolite data
    p180 = pd.read_csv("P180.csv").set_index("RID")
    nmr = pd.read_csv("NMR.csv").set_index("RID")
    dats = [p180, nmr]
    common_ids = p180.merge(nmr, how="inner", on="RID").index
    for i, dat in enumerate(dats):
        dats[i] = dat.loc[common_ids]
        plots.diagnostic_plots(dat)
        euc_dist = cdist(dat, dat, metric="euclidean")
        plots.cor_plot(
            pd.DataFrame(euc_dist),
            filename="Eucdist" + str(i),
            estimate_cor=False,
            colormap="Reds",
        )
        dats[i] = np.array(dats[i])
    Ws = snf.get_affinity_matrix(dats, 20, 0.5)
    for i, W in enumerate(Ws):
        name = "Aff" + str(i)
        np.fill_diagonal(W, 0)
        plots.cor_plot(
            pd.DataFrame(W),
            filename=name,
            estimate_cor=False,
            colormap="Reds",
        )

    # Affinity matrices are okay, but fused networks are different from R
    # But my implementation looks more similar, keep with mine
    fn = snf.SNF(Ws)
    plots.cor_plot(
        pd.DataFrame(fn),
        filename="FusedNet",
        estimate_cor=False,
        colormap="Reds",
    )
