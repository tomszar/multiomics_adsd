import itertools

import numpy as np
import pandas as pd


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
    euc_dist = np.zeros((len(dat), len(dat)))
    euc_dist = pd.DataFrame(euc_dist)
    euc_dist.index = dat.index
    euc_dist.columns = dat.index
    for i, j in itertools.combinations(dat.index, 2):
        d_ij = np.linalg.norm(dat.loc[i] - dat.loc[j])
        euc_dist.loc[i, j] = d_ij
        euc_dist.loc[j, i] = d_ij

    return euc_dist


def _affinity_matrix(mat, K, eps):
    Machine_Epsilon = np.finfo(float).eps
    Diff_mat = (mat + mat.transpose()) / 2
    Diff_mat_sort = Diff_mat - np.diag(np.diag(Diff_mat))
    Diff_mat_sort = np.sort(Diff_mat_sort, axis=0)

    K_dist = np.mean(Diff_mat_sort[:K], axis=0)
    epsilon = (K_dist + K_dist.transpose()) / 3 * 2 + Diff_mat / 3 + Machine_Epsilon

    W = np.exp(-(Diff_mat / (eps * epsilon)))

    return (W + W.transpose()) / 2
