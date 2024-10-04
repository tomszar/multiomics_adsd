import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    scaler = StandardScaler()
    for i, dat in enumerate(dats):
        dats[i] = dat.loc[common_ids]
        dats[i] = scaler.fit_transform(dats[i])
        dats[i] = np.array(dats[i])
    Ws = snf.get_affinity_matrix(dats, 20, 0.5)

    # Affinity matrices are okay, but fused networks are different from R
    # But my implementation looks more similar, keep with mine
    fn = snf.SNF(Ws)
