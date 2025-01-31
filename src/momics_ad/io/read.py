import pandas as pd

# Constants for default filenames
DEFAULT_METABOLOMICS_FILES = [
    "P180.csv",
    "NMR.csv",
    "ADNI_adnimerge_20170629_QT-freeze.csv",
]
DEFAULT_XSCORES_FILES = [
    "Xscores.csv",
    "ADNI_adnimerge_20170629_QT-freeze.csv",
]
DEFAULT_SPECTRAL_FILES = [
    "Spectral.csv",
    "ADNI_adnimerge_20170629_QT-freeze.csv",
]


def read_metabolomics(filenames: None | list[str] = None) -> dict[str, pd.DataFrame]:
    """
    Read and clean metabolomics files and return a dictionary of dataframes.

    Parameters
    ----------
    filenames : Union[None, list[str]]
        Names of files to read. If None, use default filenames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of metabolomics dataframes with keys 'p180', 'nmr', 'qt'.
    """
    filenames = filenames or DEFAULT_METABOLOMICS_FILES
    keys = ["p180", "nmr", "qt"]
    dats = _read_and_merge_files(filenames, keys)
    return dats


def read_xscores(filenames: None | list[str] = None) -> pd.DataFrame:
    """
    Read X scores from PLS-DA analysis.

    Parameters
    ----------
    filenames : Union[None, list[str]]
        Names of files to read. If None, use default filenames.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with X scores.
    """
    filenames = filenames or DEFAULT_XSCORES_FILES
    return _read_and_merge_files(filenames)


def read_spectral(filenames: None | list[str] = None) -> pd.DataFrame:
    """
    Read spectral embedding data.

    Parameters
    ----------
    filenames : Union[None, list[str]]
        Names of files to read. If None, use default filenames.

    Returns
    -------
    pd.DataFrame
        Merged dataframe of spectral embedding scores.
    """
    filenames = filenames or DEFAULT_SPECTRAL_FILES
    return _read_and_merge_files(filenames)


def _read_and_merge_files(filenames: list[str], keys: None | list[str] = None) -> dict | list:
    """
    Read and merge data from specified files. Filters rows to retain shared indices.

    Parameters
    ----------
    filenames : list[str]
        List of file paths to read and process.
    keys : Union[None, list[str]]
        Optional keys for returned dictionary. If None, return a list of dataframes.

    Returns
    -------
    Union[dict, list]
        Dictionary or list of processed dataframes depending on 'keys'.
    """
    data_list = []
    for file in filenames:
        if "QT" in file:
            data = _read_qt(file)
        else:
            data = pd.read_csv(file).set_index("RID")
        data_list.append(data)

    merged = data_list[0]
    for data in data_list[1:]:
        merged = merged.merge(data, how="inner", on="RID")

    common_ids = merged.index
    for i, data in enumerate(data_list):
        data_list[i] = data.loc[common_ids, :]

    if keys:
        return dict(zip(keys, data_list))
    else:
        return merged


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
