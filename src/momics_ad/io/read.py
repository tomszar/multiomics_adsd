import pandas as pd


def read_metabolomics(
    file_names: None | list[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Read clean metabolomics files.

    Parameters
    ----------
    file_names: Union[None, list[str]]
        Name of files. If None, use default file names from metabo_adni.
        Default None.

    Returns
    -------
    metabolites: dict[str, pd.DataFrame]
        Dictionary of dataframe of metabolite concentration, diagnosis,
        and sex. Dictionary keys are p180, nmr, and qt.
    """
    dats = {}
    keys = ["p180", "nmr", "qt"]
    if file_names is None:
        file_names = [
            "P180.csv",
            "NMR.csv",
            "ADNI_adnimerge_20170629_QT-freeze.csv",
        ]
    for i, file in enumerate(file_names):
        if "QT" in file:
            dat = _read_qt(file)
        else:
            dat = pd.read_csv(file).set_index("RID")
        dats[keys[i]] = dat
    merged = dats["p180"].merge(dats["nmr"], how="inner", on="RID")
    merged = merged.merge(dats["qt"], how="inner", on="RID")
    common_ids = merged.index
    for key in dats:
        dats[key] = dats[key].loc[common_ids, :]
    metabolites = dats
    return metabolites


def read_xscores(file_names: None | list[str] = None) -> pd.DataFrame:
    """
    Read X scores obtained from pls_da analysis.

    Parameters
    ----------
    file_names: Union[None, list[str]]
        Name of files. If None, use default file names.

    Returns
    -------
    x_scores: pd.DataFrame
        Data frame with the X scores, diagnosis, and sex.
    """
    dats = []
    if file_names is None:
        file_names = ["Xscores.csv", "ADNI_adnimerge_20170629_QT-freeze.csv"]
    for file in file_names:
        if "QT" in file:
            dat = _read_qt(file)
        else:
            dat = pd.read_csv(file).set_index("RID")
        dats.append(dat)
    x_scores = dats[0].merge(dats[1], how="inner", on="RID")
    return x_scores


def read_spectral(file_names: None | list[str] = None) -> pd.DataFrame:
    """
    Read spectral data, generated from the snf command.

    Parameters
    ----------
    file_names: Union[None, list[str]]
        Name of files. If None, use default file names.

    Returns
    -------
    spectral: pd.DataFrame
        Data frame with the spectral embedding scores, diagnosis, and sex.
    """
    dats = []
    if file_names is None:
        file_names = ["Spectral.csv", "ADNI_adnimerge_20170629_QT-freeze.csv"]
    for file in file_names:
        if "QT" in file:
            dat = _read_qt(file)
        else:
            dat = pd.read_csv(file).set_index("RID")
        dats.append(dat)
    spectral = dats[0].merge(dats[1], how="inner", on="RID")
    return spectral


def _read_qt(file: str) -> pd.DataFrame:
    """
    Read QT file data and return baseline data on diagnosis and sex.

    Parameters
    ----------
    file: str
        Name of the QT file.

    Returns
    -------
    qt: pd.DataFrame
        Data frame from QT file with RID, diagnosis, sex, baseline.
    """
    qt = pd.read_csv(
        file,
        usecols=["RID", "DX", "VISCODE", "PTGENDER"],
    ).set_index("RID")
    qt = qt.loc[qt.loc[:, "VISCODE"] == "bl", ["DX", "PTGENDER"]]
    qt = qt.loc[qt["DX"].isin(["NL", "MCI", "Dementia"]), :]
    return qt
