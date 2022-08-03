import argparse
import pandas as pd
from .io import read
from .stats import multi


def main():
    '''
    The main routine
    '''
    parser = argparse.ArgumentParser(
        description='Multivariate multiomics sex differences')
    parser.add_argument('-R',
                        type=int,
                        default=30,
                        metavar='REPEATS',
                        help='Number of times the CV loop should be repeated.')
    parser.add_argument('-C',
                        type=int,
                        default=50,
                        metavar='COMPONENTS',
                        help='Maximum number of components to evaluate.')
    args = parser.parse_args()
    m = read.read_files()
    X = m.iloc[:, :-1]
    Y = m.iloc[:, -1]
    model_table = multi.plsda_doubleCV(X, Y, n_repeats=args.R,
                                       max_components=args.C)
    i_b_m = model_table['table'].iloc[:, 2].idxmax()
    best_mod = model_table['models'][i_b_m]
    xscores = pd.DataFrame(best_mod.transform(X))
    xscores.to_csv('Xscores.csv')
    model_table['table'].to_csv('ModelTable.csv')
