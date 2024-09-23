import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momics_ad.stats import sd


def scatter_plot(scores: pd.DataFrame):
    """
    Scatter plot from all dimensions, divided by sex,
    and diagnosis highlighted.

    Parameters
    ----------
    scores: pd.DataFrame
        Dataframe with scores, sex, and diagnosis columns,
        as obtained from the pls_da command.
    """
    last_col = scores.columns.get_loc("DX") - 1
    # Get mean coordinates
    X = scores.loc[:, ["DX", "PTGENDER"]]
    Y = scores.iloc[:, : last_col + 1]
    model_full = sd.get_model_matrix(X)
    betas = sd.estimate_betas(model_full, Y)
    ls_matrix = sd._get_ls_vectors()
    means = np.matmul(ls_matrix, betas)
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
        "Female Control": means.iloc[0, :],
        "Female MCI": means.iloc[1, :],
        "Female AD": means.iloc[2, :],
        "Male Control": means.iloc[3, :],
        "Male MCI": means.iloc[4, :],
        "Male AD": means.iloc[5, :],
    }
    how_many_axes = int((last_col + 1) / 2)
    if how_many_axes == 1:
        how_many_cols = 1
        how_many_rows = 1
    else:
        how_many_cols = 2
        how_many_rows = math.ceil(how_many_axes / how_many_cols)

    fig = plt.figure(figsize=(12, how_many_rows * 6))
    spec = fig.add_gridspec(ncols=how_many_cols, nrows=how_many_rows)
    offset = 0.5
    plot_number = 1
    ax_col = 0
    ax_row = 0
    looping = True
    while looping:
        for idx1 in range(0, last_col + 1, 2):
            ax = fig.add_subplot(spec[ax_row, ax_col])
            # Set manual limits
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for main_key, main_value in label_main_colors.items():
                ax.scatter(
                    scores.loc[scores["PTGENDER"] == main_key, str(idx1)],
                    scores.loc[scores["PTGENDER"] == main_key, str(idx1 + 1)],
                    color=main_value,
                    alpha=0.1,
                )
                if main_key == "Female":
                    ax.plot(
                        means.iloc[:3, idx1],
                        means.iloc[:3, idx1 + 1],
                        color=main_value,
                    )
                else:
                    ax.plot(
                        means.iloc[3:, idx1],
                        means.iloc[3:, idx1 + 1],
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
    fig.savefig("ScatterPlot.pdf", dpi=600)


def cor_plot(dat: pd.DataFrame):
    """
    Generate a correlation plot out of multiple variables.

    Parameters
    ----------
    dat: pd.DataFrame
        Data frame with the variables to generate a correlation plot.
    """
    fig, ax = plt.subplots(figsize=(28, 28))
    cor_matrix = dat.corr()
    ax.imshow(cor_matrix, cmap="RdBu")
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
    fig.savefig("CorPlot.pdf", dpi=200)
