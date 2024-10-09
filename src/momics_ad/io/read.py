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
            dat = pd.read_csv(
                file,
                usecols=["RID", "DX", "VISCODE", "PTGENDER"],
            ).set_index("RID")
            dat = dat.loc[dat.loc[:, "VISCODE"] == "bl", ["DX", "PTGENDER"]]
            dat = dat.loc[dat["DX"].isin(["NL", "MCI", "Dementia"]), :]
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
    file_name: Union[None, list[str]]
        Name of files. If None, use default file names.

    Returns
    -------
    x_scores: pd.DataFrame
        Data frame with the X scores, diagnosis, and sex.
    """
    dats = []
    if file_names is None:
        file_names = ["Xscores.csv", "ADNI_adnimerge_20170629_QT-freeze.csv"]
    for i, file in enumerate(file_names):
        if "QT" in file:
            dat = pd.read_csv(
                file, usecols=["RID", "DX", "VISCODE", "PTGENDER"]
            ).set_index("RID")
            dat = dat.loc[dat.loc[:, "VISCODE"] == "bl", ["DX", "PTGENDER"]]
        else:
            dat = pd.read_csv(file).set_index("RID")
        dats.append(dat)
    x_scores = dats[0].merge(dats[1], how="inner", on="RID")
    return x_scores
