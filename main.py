#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats

plt.rcParams["figure.figsize"] = (8, 8)


def read_in_data(fpath):
    """
    Read in data into a manipulatable data format.

    Arguments:
        fpath str
            Path to a csv with ";" delimeted columns. Should contain column headers
            including (swim, bike, run, finish, category) as if from SportStats.

    Returns:
        df pd.DataFrame
            A Pandas DataFrame object containing
    """
    df = pd.read_csv(fpath, sep=";")
    df = df.rename({x: x.lower() for x in df.columns}, axis=1).dropna(subset=["finish"])

    df["sex"] = df.category.str[0]
    df = df.apply(_str_timedelta_to_float_min, axis=0)

    return df.set_index("bib")


def _str_timedelta_to_float_min(ser):
    """
    Attempts to cast a timedelta series to float value in minutes.

    Argument:
        ser pd.Series
            A Pandas Series object.

    Returns
        ser pd.Series
            A Pandas Series object.
    """
    if ser.dtype is not np.dtype("O"):
        return ser
    try:
        return pd.to_timedelta(ser, unit="m").dt.seconds / 60
    except:
        return ser


def plot_discipline(df, bib_numbers, title=None):
    """
    Plot the KDE and shaded percentile of bib numbers for each triathlon discipline.

    Arguments
        df pd.DataFrame
            A pandas dataframe guarenteed to have columns
                 [swim, bike, run, finish, sex]
            Indexed on bib numbers
        bib_numbers tuple(int)
            A tuple of bib numbers of interest. Only one for each gender is supported.
        title str | None
            An optional title for the plot
    """
    fig = plt.figure()

    ncols, nrows = 2, 2
    disciplines = ["swim", "bike", "run", "finish"]
    for i, discipline in enumerate(disciplines):
        plt.subplot(ncols, nrows, i + 1)
        plt.title(discipline[0].upper() + discipline[1:])
        plt.yticks([])

        plot_kde(
            data=df,
            col=discipline,
            hue="sex",
            shade_right_idx=bib_numbers,
            legend=False,
        )

    fig.axes[1].legend(loc=(1.05, 0.835))
    plt.text(
        1,
        -0.15,
        "Minutes",
        horizontalalignment="left",
        verticalalignment="center",
        transform=fig.axes[2].transAxes,
    )
    plt.text(
        -0.1,
        0.25,
        "Probability Density Functions",
        horizontalalignment="center",
        verticalalignment="baseline",
        transform=fig.axes[0].transAxes,
        rotation=90,
    )

    if title is not None:
        plt.suptitle(title)

    plt.savefig("png/triathlon.png", bbox_inches="tight", pad=0)


def plot_kde(data, col, hue, shade_right_idx, legend=False):
    """
    Plot a KDE along with a shaded percentile.

    Additionally print the shaded percentile value.

    Arguments
        df pd.DataFrame
            A pandas dataframe.
        col str
            The DataFrame column to construct a KDE with.
        hue str
            Column by which to group the data frame and create a KDE for each with.
        shade_right_idx tuple(int)
            A tuple of indices to calculate percentiles and shade right of for a KDE.
        legend bool
            Whether or not to plot a legend.
    """
    for group, df in data.groupby(hue):
        observations = df[col].rename(group).dropna()

        shade_right = _switch_shade_value(df, col, shade_right_idx)
        if shade_right is None:
            continue

        kde = scipy.stats.gaussian_kde(observations)
        domain = (shade_right, 1.1 * observations.max())
        percentile = kde(np.arange(*domain, 1)).sum()

        plot_percentile(
            observations=observations, shade_right=shade_right, legend=legend
        )
        print(f"{group}\t{col}\t{percentile}")


def _switch_shade_value(df, col, key_idx):
    """
    Houses logic for identifying one unique value from a DataFrame.

    Arguments
        df pd.DataFrame
            A pandas dataframe.
        col str
            A column of a pandas DataFrame to return the indexed value of.
        key_idx tuple(int)
            A tuple of indices.

    Returns
        None | object
            Value corresponding to one and only one valid result from a set of potential
            indices.

    Raises
        ValueError
            Thrown in the ambiguous case of multiple indice values which are possible.
    """
    ser = df.reindex(key_idx)[col].dropna()
    if ser.shape[0] == 0:
        return None
    elif ser.shape[0] == 1:
        return ser.values[0]
    else:
        raise ValueError("Multiple indices found.")


def plot_percentile(observations, shade_right=None, legend=False):
    """
    Plot the kde of a set of observations along with a shade percentile.

    Arguments
        observations list(np.numeric)
            A list of numeric observations to construct a KDE around.
        shade_right None | np.numeric
            Whether or not to additinoally shade the region right of this value.
        legend bool
            Whether or not to plot a legend.
    """
    sns.kdeplot(observations, shade=True, legend=legend)

    if shade_right is None:
        return

    kde = scipy.stats.gaussian_kde(observations)
    domain = np.arange(shade_right, plt.xlim()[1])
    plt.fill_between(domain, kde(domain), alpha=0.5)


def main(fpath, bib_numbers):
    """
    Main interface with CLI.

    Arguments:
        fpath str
            Path to a csv with ";" delimeted columns. Should contain column headers
            including (swim, bike, run, finish, category) as if from SportStats.
        bib_numbers: tuple(int)
            A tuple of bib numbers to calculate percentile values of. The usual case
            would be one for male and female results. If at any point the KDEs formed
            of the constructed groups have multiple valid bib_numbers it will be deemed
            ambiguous and throw an error.
    """
    df = read_in_data(fpath)
    plot_discipline(df, bib_numbers)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description=(
            "Generate plots of percentile values occuring from a KDE and print out"
            "numeric values"
        )
    )

    PARSER.add_argument(
        "fpath",
        help=(
            'Path to a csv with ";" delimeted columns. Should contain column headers'
            "including (swim, bike, run, finish, category) as if from SportStats."
        ),
    )
    PARSER.add_argument(
        "--bib_numbers",
        nargs="*",
        type=int,
        help=(
            "Bib numbers to calculate percentile values of. The usual case would be one"
            "for male and female results. If at any point the KDEs formed of the"
            "constructed groups have multiple valid bib_numbers it will be deemed"
            "ambiguous and throw an error."
        ),
        default=[],
    )

    ARGS = PARSER.parse_args()

    main(ARGS.fpath, tuple(ARGS.bib_numbers))
