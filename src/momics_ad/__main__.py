import argparse
from .io import read


def main():
    '''
    The main routine
    '''
    parser = argparse.ArgumentParser(
        description='Multivariate multiomics sex differences')
    parser.add_argument('-P',
                        type=str,
                        default='p180',
                        metavar='PLATFORM',
                        help='Select the platform to analyze,\
                              either p180 or nmr.\
                              Default: p180.')
    args = parser.parse_args()
    dats = read.read_files(args.P)
    print(dats)
