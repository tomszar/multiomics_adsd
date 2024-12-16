import math
import os

import matplotlib.axes as axs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momics_ad.io import subset
from momics_ad.stats import sd


def scatter_plot(
    scores: pd.DataFrame,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    filename: str = "ScatterPlot.pdf",
):
    """
    Scatter plot from all dimensions, divided by sex,
    and diagnosis highlighted.

    Parameters
    ----------
    scores: pd.DataFrame
        Dataframe with scores, sex, and diagnosis columns,
        as obtained from the pls_da command.
    xlim: tuple[int, int] | None
        Lower and upper limits on the x axis. Default None
    ylim: tuple[int, int] | None
        Lower and upper limits on the y axis. Default None
    filename: str
        Filename of the plot, including extension. Default ScatterPlot.pdf
    """
    _check_dir()
    # Get mean coordinates
    X, Y = subset.get_XY(scores)
    obs_vect = sd.get_observed_vectors(X, Y)
    # Get colors and coordinates by label
    tab20b = plt.get_cmap("tab20b")
    label_main_colors = {
        "Female": tab20b(0),
        "Male": tab20b(4),
    }
    label_colors = {
        "Female Control": tab20b(1),
        "Female MCI": tab20b(2),
        "Female AD": tab20b(3),
        "Male Control": tab20b(5),
        "Male MCI": tab20b(6),
        "Male AD": tab20b(7),
    }
    label_coords = {
        "Female Control": obs_vect.iloc[0, :],
        "Female MCI": obs_vect.iloc[1, :],
        "Female AD": obs_vect.iloc[2, :],
        "Male Control": obs_vect.iloc[3, :],
        "Male MCI": obs_vect.iloc[4, :],
        "Male AD": obs_vect.iloc[5, :],
    }
    last_col = scores.columns.get_loc("DX") - 1
    how_many_axes = int((last_col + 1) / 2)
    if how_many_axes == 1:
        how_many_cols = 1
        how_many_rows = 1
    else:
        how_many_cols = 2
        how_many_rows = math.ceil(how_many_axes / how_many_cols)

    fig = plt.figure(figsize=(12, how_many_rows * 6))
    spec = fig.add_gridspec(ncols=how_many_cols, nrows=how_many_rows)
    # offset = 0.5
    plot_number = 1
    ax_col = 0
    ax_row = 0
    looping = True
    while looping:
        for idx1 in range(0, last_col + 1, 2):
            ax = fig.add_subplot(spec[ax_row, ax_col])
            # Set manual limits
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
            for main_key, main_value in label_main_colors.items():
                ax.scatter(
                    scores.loc[scores["PTGENDER"] == main_key, str(idx1)],
                    scores.loc[scores["PTGENDER"] == main_key, str(idx1 + 1)],
                    color=main_value,
                    alpha=0.1,
                )
                ax.set_xlabel("Coordiante " + str(idx1 + 1))
                ax.set_ylabel("Coordinate " + str(idx1 + 2))
                if main_key == "Female":
                    ax.plot(
                        obs_vect.iloc[:3, idx1],
                        obs_vect.iloc[:3, idx1 + 1],
                        color=main_value,
                    )
                else:
                    ax.plot(
                        obs_vect.iloc[3:, idx1],
                        obs_vect.iloc[3:, idx1 + 1],
                        color=main_value,
                    )
            for key, value in label_colors.items():
                x = label_coords[key].iloc[idx1]
                y = label_coords[key].iloc[idx1 + 1]
                ax.scatter(x, y, s=70, color=value)
                # ax.annotate(key, (x + offset, y + offset))
            if (plot_number / 2).is_integer():
                ax_row = ax_row + 1
                ax_col = 0
            else:
                ax_col = ax_col + 1
            plot_number = plot_number + 1
            if plot_number > how_many_axes:
                looping = False
    fig.tight_layout()
    fig.savefig("plots/" + filename, dpi=600)


def diagnostic_plots(dat: pd.DataFrame, name: str):
    """
    Diagnostic plots of continuous distributions.
    Originally thought to plot the metabolites values of the p180
    and nmr platforms.

    Parameters
    ----------
    dat: pd.DataFrame
        Data frame in which to plot all variables.
        Should have column names and variables are all continuous.
    name: str
        Name for the diagnostic plot file.
    """
    _check_dir()
    n_vars = dat.shape[1] + 1
    n_cols, n_rows = _how_many_plots(n_vars, False)
    fig = plt.figure()
    spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows)
    plot_number = 1
    ax_col = 0
    ax_row = 0
    looping = True
    # if n_vars > 8:  # Use multi pdf
    #    with PdfPages("diagnostic_plots/" + name + ".pdf") as pdf:
    while looping:
        for _, col in enumerate(dat):
            ax = fig.add_subplot(spec[ax_row, ax_col])
            hist_plot(dat.loc[:, col], ax)
        if plot_number > n_vars:
            looping = False
    fig.tight_layout()
    fig.savefig("diagnostic_plots/" + name + ".pdf", dpi=300)


def hist_plot(
    dat: pd.DataFrame | pd.Series,
    ax: axs.Axes,
):
    """Plot a histogram

    Parameters
    ----------
    dat: pd.DataFrame, pd.Series
        Data set or series to plot.
    ax: axs.Axes
        Axes object.
    """
    ax.hist(dat)


def orientation_plot(scores: pd.DataFrame):
    """
    Trajectory plot ordered by diagnoses (CN, MCI, AD) and
    in ordered dimensions.

    Parameters
    ----------
    scores: pd.DataFrame
        Dataframe with scores, sex, and diagnosis columns,
        as obtained from the pls_da command.
    """
    _check_dir()
    tab20b = plt.get_cmap("tab20b")
    # Get mean coordinates
    X, Y = subset.get_XY(scores)
    obs_vect = sd.get_observed_vectors(X, Y)
    contrast = [[0, 1, 2], [3, 4, 5]]
    n_groups = len(contrast)
    ys = []
    for i in range(n_groups):
        y = sd._estimate_orientation(obs_vect, contrast[i])
        ys.append(y)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    ax.plot(ys[0], color=tab20b(0))
    ax.plot(ys[1], color=tab20b(4))
    ax.set_xlabel("Coordinates")
    ticks = list(range(obs_vect.shape[1]))
    labels = [str(i + 1) for i in ticks]
    ax.set_xticks(ticks=ticks, labels=labels)
    fig.tight_layout()
    fig.savefig("plots/OrientationPlot.pdf", dpi=600)


def cor_plot(
    dat: pd.DataFrame,
    filename: str = "CorPlot",
    estimate_cor: bool = True,
    colormap: str = "coolwarm",
):
    """
    Generate a correlation plot out of multiple variables.

    Parameters
    ----------
    dat: pd.DataFrame
        Data frame with the variables to generate a correlation plot.
    filename: str
        Name of the file to use. Default CorPlot.
    estimate_cor: bool
        Whether to estimate the correlation from the dat data frame.
        If False, it assumes the correlation, or similar metric, is
        already used in dat. Default True.
    colormap: str
        Colormap to use. Default coolwarm.
    """
    _check_dir()
    fig, ax = plt.subplots(figsize=(28, 28))
    if estimate_cor:
        cor_matrix = dat.corr()
    else:
        cor_matrix = dat
    ax.imshow(
        cor_matrix,
        cmap=colormap,
        norm="log",
    )
    ax.set_xticks(np.arange(len(cor_matrix)), labels=cor_matrix.index)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    plt.tick_params(
        which="both",  # both major and minor ticks are affected
        left=False,
        right=False,
        labelleft=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig("plots/" + filename + ".pdf", dpi=200)


def vip_plot(dat: pd.DataFrame, fnames: list[str]):
    """
    Generate a Manhattan Plot style with VIPs.

    Parameteres
    -----------
    dat: pd.DataFrame
        VIP dataframe as generated from pls_da command.
    fnames: list[str]
        List of feature names.
    """
    _check_dir()
    fig, ax = plt.subplots(figsize=(60, 6))
    ax.scatter(y=dat.iloc[:, 0], x=fnames)  # list(range(len(dat))))
    ax.hlines(1.2, xmin=0, xmax=len(dat), colors="r", linestyles="dashed")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig("plots/VIPPlot.pdf", dpi=300)


def plot_embedding(embedding: np.ndarray, qt: pd.DataFrame):
    """
    Plot spectral clustering embedding.

    Parameters
    ----------
    embedding: np.ndarray
        Embedding matrix.
    qt: pd.DataFrame
        Dataframe with Sex and diagnosis data.
    """
    _check_dir()
    for col in qt:
        fig, ax = plt.subplots()
        color = list(qt[col])
        for c in set(color):
            keep = qt[col] == c
            ax.scatter(embedding[keep, 0], embedding[keep, 1])
        fig.tight_layout()
        fig.savefig("plots/Spectral" + col + ".pdf", dpi=300)
        plt.close()


def _check_dir():
    """
    Check and create directory for plots if not exists.
    """
    for folder in ["plots", "diagnostic_plots"]:
        os.makedirs(folder, exist_ok=True)


def _how_many_plots(n_vars: int, pairwise: bool) -> tuple[int, int]:
    """
    Obtain the number of rows and columns to use to plot
    a given the number of axes. If n_vars is odd and pairwise
    true, the number of axes does not count the last variable.

    Parameters
    ----------
    n_vars: int
        Number of variables to plot.
    pairwise: bool
        Whether the plots are intended for paiwise data.

    Returns
    -------
    n_cols, n_rows: tuple[int, int]
        Number of columns and rows to use.
    """
    # Getting number of axes
    if pairwise:
        n_axes = int(n_vars / 2)
    else:
        n_axes = n_vars
    # Getting number of cols
    if n_axes == 1:
        n_cols = 1
    else:
        n_cols = 2
    # Get number of rows
    n_rows = math.ceil(n_axes / 2)
    return n_cols, n_rows
