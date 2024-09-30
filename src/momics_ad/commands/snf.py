import pandas as pd
import snf
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

from momics_ad.stats.snf import _affinity_matrix, _euclidean_dist, _sparse_kernel


def main():
    """
    Main SNF routine.
    """
    # Read metabolite data
    p180 = pd.read_csv("P180.csv").set_index("RID")
    nmr = pd.read_csv("NMR.csv").set_index("RID")
    dats = [p180, nmr]
    common_ids = p180.merge(nmr, how="inner", on="RID").index
    scaler = StandardScaler()
    for i, dat in enumerate(dats):
        dats[i] = dat.loc[common_ids]
        dats[i] = scaler.fit_transform(dats[i])

    # Calculate distance
    distance = cdist(dats[0], dats[0], metric="euclidean")
    distance2 = pdist(dats[0], metric="euclidean")
    distance3 = _euclidean_dist(pd.DataFrame(dats[0]))

    Ws = snf.make_affinity(
        dats,
        metric="euclidean",
        K=20,
        mu=0.5,
    )
    # All two affinity is different from the one in R,
    aff = _affinity_matrix(distance, 20, 0.5)

    fused_network = snf.snf(Ws, K=20)
