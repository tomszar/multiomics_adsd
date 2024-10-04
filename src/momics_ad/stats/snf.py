import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist


def SNF(Ws: list[np.ndarray], k: int = 20, t: int = 20) -> np.ndarray:
    """
    Similarity network fusion for t iterations using k nearest neighbors.

    Parameters
    ----------
    Ws: list[np.ndarray]
        List of matrices to apply the similarit network fusion.
    k: int
        Number of nearest neighbors to use. Default 20.
    t: int
        Number of iterations to run SNF. Default 20.

    Returns
    -------
    Pc: np.ndarray
        Fused matrix.
    """
    nw = len(Ws)
    Ps = []
    Ss = []

    for i in range(nw):
        Ps.append(_full_kernel(Ws[i]))
        Ps[i] = (Ps[i] + Ps[i].transpose()) / 2
        Ss.append(_sparse_kernel(Ps[i], k))

    # Generate to Ps that will be updated
    Pst0 = Ps.copy()
    Pst1 = Ps.copy()

    # Iteration process
    for i in range(t):
        for j in range(nw):
            if j == 0:
                m = 1
            elif j == 1:
                m = 0
            # TODO: m is definitely unbound,
            # because we can expect more than two nw
            Pst1[j] = np.matmul(np.matmul(Ss[j], Pst0[m]), Ss[j].transpose())
            Pst1[j] = _full_kernel(Pst1[j])
        # Update the zero state
        Pst0 = Pst1.copy()

    Pc = (Pst1[0] + Pst1[1]) / 2
    return Pc


def get_affinity_matrix(
    dats: list[np.ndarray], K: int = 20, eps: float = 0.5
) -> list[np.ndarray]:
    """
    Estimate the affinity matrix for all datasets in dats from the squared Euclidean
    distance.

    Parameters
    ----------
    dats: list[np.ndarray]
        list of data sets to estimate the affinity matrix.
    K: int
        Number of K nearest neighbors to use. Default 20.
    eps: float
        Normalization factor. Recommended between 0.3 and 0.8. Default 0.5.

    Returns
    -------
    Ws: list[np.ndarray]
        list of affinity matrices
    """
    nrows = len(dats[0])
    Ws = [np.zeros((nrows, nrows)), np.zeros((nrows, nrows))]
    for i, dat in enumerate(dats):
        euc_dist = cdist(dat, dat, metric="euclidean") ** 2
        Ws[i] = _affinity_matrix(euc_dist, K, eps)

    return Ws


def _full_kernel(W: np.ndarray) -> np.ndarray:
    """
    Calculate full kernel matrix normalization.

    Parameters
    ----------
    W: np.ndarray
        Matrix in which to perform the full kernel normalization.

    Returns
    -------
    P: np.ndarray
        Normalized matrix.
    """

    rowsum = W.sum(axis=1) - W.diagonal()
    rowsum[rowsum == 0] = 1
    P = W / (2 * rowsum)

    np.fill_diagonal(P, 0.5)
    return P


def _sparse_kernel(W: np.ndarray, k: int) -> np.ndarray:
    """
    Calculate sparse kernel matrix using the k nearest neighbors.

    Parameters
    ----------
    W: np.ndarray
        Matrix in which to perform the sparse kernel matrix.
    k: int
        Number of nearest neighbors to use.

    Returns
    -------
    S: np.ndarray
        Sparse kernel matrix.
    """
    nrow = len(W)
    S = np.zeros((nrow, nrow))
    for i in range(0, nrow):
        s1 = W[i, :].copy()
        ix = s1.argsort()
        last = nrow - k
        s1[ix[0:last]] = 0
        S[i,] = s1

    S = _full_kernel(S)
    return S


def _euclidean_dist(dat: pd.DataFrame) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distance between
    all the rows of a dataframe
    """
    euc_dist = cdist(dat, dat, metric="euclidean")
    euc_dist.index = dat.index
    euc_dist.columns = dat.index

    return euc_dist


def _affinity_matrix(mat, K, eps):
    Machine_Epsilon = np.finfo(float).eps
    Diff_mat = (mat + mat.transpose()) / 2
    # Sort distance matrix ascending order
    # (i.e. more similar is closer to first column)
    Diff_mat_sort = Diff_mat - np.diag(np.diag(Diff_mat))
    Diff_mat_sort = np.sort(Diff_mat_sort, axis=1)
    # Average distance with K nearest neighbors
    K_dist = np.mean(Diff_mat_sort[:, 1 : (K + 1)], axis=1) + Machine_Epsilon
    sigma = ((np.add.outer(K_dist, K_dist) + Diff_mat) / 3) + Machine_Epsilon
    sigma[sigma < Machine_Epsilon] = Machine_Epsilon

    W = stats.norm.pdf(Diff_mat, loc=0, scale=(eps * sigma))

    return (W + W.transpose()) / 2
