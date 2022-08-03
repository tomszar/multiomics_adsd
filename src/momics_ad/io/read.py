import pandas as pd
from typing import Union


def read_files(file_names: Union[None, list[str]] = None) ->\
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
        Dataframe of metabolite concentration.
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
