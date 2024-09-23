import pandas as pd


def get_XY(dat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split score data into an X and Y matrix.

    Parameters
    ----------
    dat: pd.DataFrame
        Score data as obtined from the pls_da

    Returns
    -------
    X: pd.DataFrame
        X matrix of responses.
    Y: pd.DataFrame
        Y matrix of outcomes.
    """
    last_col = str(dat.columns.get_loc("DX") - 1)
    X = dat.loc[:, ["DX", "PTGENDER"]]
    Y = dat.loc[:, :last_col]
    return X, Y
