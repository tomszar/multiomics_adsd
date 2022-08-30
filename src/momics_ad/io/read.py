import pandas as pd
from typing import Union


def read_metabolomics(file_names: Union[None, list[str]] = None) ->\
        pd.DataFrame:
    '''
    Read clean metabolomics files.

    Parameters
    ----------
    file_names: Union[None, list[str]]
        Name of files. If None, use default file names from metabo_adni.

    Returns
    -------
    metabolites: pd.DataFrame
        Dataframe of metabolite concentration, diagnosis, and sex.
    '''
    dats = []
    if file_names is None:
        file_names = ['P180.csv',
                      'NMR.csv',
                      'ADNI_adnimerge_20170629_QT-freeze.csv']
    for i, file in enumerate(file_names):
        dat = pd.read_csv(file).set_index('RID')
        if 'QT' in file:
            dat = dat.loc[dat.loc[:, 'VISCODE'] == 'bl', 'DX']
            dat = dat[dat.isin(['NL', 'MCI', 'Dementia'])]
        dats.append(dat)

    metabolites = dats[0].merge(dats[1],
                                how='inner',
                                on='RID')
    metabolites = metabolites.merge(dats[2],
                                    how='inner',
                                    on='RID')
    return metabolites


def read_xscores(file_names: Union[None, list[str]] = None) -> pd.DataFrame:
    '''
    Read X scores obtained from pls_da analysis.

    Parameters
    ----------
    file_name: Union[None, list[str]]
        Name of files. If None, use default file names.

    Returns
    -------
    x_scores: pd.DataFrame
        Data frame with the X scores, diagnosis, and sex.
    '''
    dats = []
    if file_names is None:
        file_names = ['Xscores.csv',
                      'ADNI_adnimerge_20170629_QT-freeze.csv']
    for i, file in enumerate(file_names):
        dat = pd.read_csv(file).set_index('RID')
        if 'QT' in file:
            dat = dat.loc[dat.loc[:, 'VISCODE'] == 'bl', ['DX', 'PTGENDER']]
        dats.append(dat)
    x_scores = dats[0].merge(dats[1],
                             how='inner',
                             on='RID')
    return x_scores
