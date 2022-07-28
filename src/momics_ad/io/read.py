import pandas as pd
from typing import Union
from metabo_adni.data import load


def read_files(platform: str,
               file_names: Union[None, list[str]] = None) ->\
        pd.DataFrame:
    '''
    Read clean metabolomics files.

    Parameters
    ----------
    platform: str
        Platform to read, either p180 or nmr.
    file_names: Union[None, list[str]]
        Name of files. If None, use default file_names.

    Returns
    -------
    metabolites: pd.DataFrame
        Dataframe of metabolite concentration.
    '''
    dats = []
    if file_names is None:
        file_names = get_filenames(platform)
    for i, file in enumerate(file_names):
        dat = pd.read_csv(file).set_index('RID')
        col_names = load._get_metabo_col_names(dat,
                                               file_names[i])
        dat = dat.loc[:, col_names]
        dats.append(dat)

    if platform == 'nmr':
        metabolites = pd.DataFrame(dats)
    elif platform == 'p180':
        merge1 = dats[0].merge(dats[1],
                               how='inner',
                               on='RID',
                               suffixes=('_1UPLC', '_1FIA'))
        merge2 = dats[2].merge(dats[3],
                               how='inner',
                               on='RID',
                               suffixes=('_2UPLC', '_2FIA'))
        metabolites = pd.concat([merge1, merge2])
    else:
        metabolites = pd.DataFrame()
    return metabolites


def get_filenames(platform: str) -> list[str]:
    '''
    Get list of filenames based on platform.

    Parameters
    ----------
    platform: str
        Platform to read, either p180 or nmr.

    Returns
    -------
    file_names: list[str]
        List of filenames.
    '''
    if platform == 'p180':
        file_names = ['ADNI1-UPLC.csv',
                      'ADNI1-FIA.csv',
                      'ADNI2GO-UPLC.csv',
                      'ADNI2GO-FIA.csv']
    elif platform == 'nmr':
        file_names = ['NMR.csv']
    else:
        file_names = []

    return file_names
