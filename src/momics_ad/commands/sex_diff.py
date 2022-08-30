import argparse
import numpy as np
import pandas as pd
from momics_ad.io import read
from momics_ad.stats import sd


def main():
    '''
    Sex difference analysis main routine.
    '''
    parser = argparse.ArgumentParser(
        description='Sex difference analysis.')
    parser.add_argument('-I',
                        type=int,
                        default=999,
                        metavar='ITERATIONS',
                        help='Number of iterations to run the randomization.\
                        Default 999.')
    args = parser.parse_args()
    x_scores = read.read_xscores()
    x_center = sd.center_matrix(x_scores)
    last_col = str(x_center.columns.get_loc('DX') - 1)
    X = x_center.loc[:, ['DX', 'PTGENDER']]
    Y = x_center.loc[:, :last_col]
    model_full = sd.get_model_matrix(X)
    model_red = sd.get_model_matrix(X, full=False)
    contrast = [[0, 1, 2], [3, 4, 5]]
    # Estimate LS vectors
    f_c = [1, 0, 0, 0, 0, 0]
    f_m = [1, 0, 1, 0, 0, 0]
    f_d = [1, 1, 0, 0, 0, 0]
    m_c = [1, 0, 0, 1, 0, 0]
    m_m = [1, 0, 1, 1, 0, 1]
    m_d = [1, 1, 0, 1, 1, 0]
    x_ls_full = np.array([f_c, f_m, f_d,
                          m_c, m_m, m_d])
    delta, angle, shape = sd.estimate_difference(Y, model_full,
                                                 x_ls_full,
                                                 contrast)
    deltas, angles, shapes = sd.RRPP(Y, model_full,
                                     model_red,
                                     x_ls_full,
                                     contrast,
                                     args.I)
    total_rep = args.I + 1
    pvals = [(sum(angles > angle) / total_rep)[0, 1],
             (sum(deltas > delta) / total_rep)[0, 1],
             (sum(shapes > shape) / total_rep)[0, 1]]
    vals = {'Values': [delta[0, 1], angle[0, 1], shape[0, 1]],
            'Index': ['delta', 'angle', 'shape'],
            'Pvalues': pvals}
    result_table = pd.DataFrame(vals).set_index('Index')
    result_table.to_csv('ResultTable.csv')
