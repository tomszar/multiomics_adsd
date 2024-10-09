import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from momics_ad.figures import plots
from momics_ad.io import read
from momics_ad.stats import snf


def main():
    """
    Main SNF routine.
    """
    # Read metabolite data
    mets = read.read_metabolomics()
    labels = ["p180", "nmr"]
    mets_array = []
    for key in mets:
        if key == "qt":
            pass
        else:
            dat = mets[key]
            # plots.diagnostic_plots(dats[i], name="Diagnostic" + str(i))
            euc_dist = cdist(dat, dat, metric="euclidean")
            plots.cor_plot(
                pd.DataFrame(euc_dist),
                filename="Eucdist" + key,
                estimate_cor=False,
                colormap="Reds",
            )
            mets_array.append(np.array(dat))
    Ws = snf.get_affinity_matrix(mets_array, 20, 0.5)
    for i, W in enumerate(Ws):
        name = "Aff" + labels[i]
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
    print("embedding")
    embedding = snf.get_spectral(fn)
    plots.plot_embedding(embedding)
    embedding = pd.DataFrame(embedding)
    embedding.index = mets["p180"].index
    pd.DataFrame(embedding).to_csv("Spectral.csv")
