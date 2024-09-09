import matplotlib.pyplot as plt
import pandas as pd


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
    dark2 = plt.get_cmap("Dark2")
    label_colors = {"Female": dark2(0), "Male": dark2(1)}
    last_col = scores.columns.get_loc("DX") - 1
    how_many_axes = int((last_col + 1) / 2)
    if how_many_axes == 1:
        how_many_cols = 1
        how_many_rows = 1
    else:
        how_many_cols = 2
        how_many_rows = round(how_many_axes / how_many_cols)

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
            for key, value in label_colors.items():
                means = (
                    scores.loc[scores["PTGENDER"] == key, "0":"DX"].groupby("DX").mean()
                )
                ax.scatter(
                    scores.loc[scores["PTGENDER"] == key, str(idx1)],
                    scores.loc[scores["PTGENDER"] == key, str(idx1 + 1)],
                    color=value,
                    alpha=0.3,
                )
                ax.plot(
                    means.loc[:, str(idx1)],
                    means.loc[:, str(idx1 + 1)],
                    color=value,
                )
                for i, rows in means.iterrows():
                    x = means.loc[i, str(idx1)]
                    y = means.loc[i, str(idx1 + 1)]
                    ax.scatter(
                        x,
                        y,
                        s=60,
                        color=value,
                    )
                    ax.annotate(i, (x + offset, y + offset))
            if (plot_number / 2).is_integer():
                ax_row = ax_row + 1
            else:
                ax_col = ax_col + 1
            plot_number = plot_number + 1
            if plot_number > how_many_axes:
                looping = False
    fig.tight_layout()
    fig.savefig("ScatterPlot.pdf", dpi=600)
    plt.show()
