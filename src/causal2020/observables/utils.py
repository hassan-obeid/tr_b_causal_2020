"""
Utility functions used in selection on observables work
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_params(sim_par, model, fig_size):
    """
    Function to plot parameters from simulated data.

    Parameters
    ----------
    sim_par: Pandas Series or Pandas Column
        Pandas Series or column from DataFrame containing
        coefficients from the estimated models based on
        simulated data.

    model: Pylogit model.
        Estimated data generating model to compare
        estimated parameters based on simulated data
        to.

    size: tuple
        Figure size

    Returns
    -------
    Seaborn figure of the distribution of estimated parameters
    based on simulated data.
    """
    # Create bins for the histogram
    bins = np.linspace(sim_par.min(), sim_par.max(), 10)

    # Initialize the figure size
    plt.figure(figsize=fig_size)

    # Plot the distribution
    sns.distplot(sim_par, kde=False, bins=bins, label="Simulated Parameters")

    # Add vertical line for the data generating parameter
    plt.axvline(
        model.coefs[sim_par.name],
        color="black",
        ls="--",
        label="Data Generating Parameter",
    )

    # Add a title for the figure
    plt.title(
        label="Histogram of " + '"' + sim_par.name + '"',
        fontdict={"fontsize": 16},
    )

    # Add a y-label
    plt.ylabel("Frequency", rotation=0, labelpad=50)

    # Add a legend
    plt.legend()


def find_outliers(data, threshold=3.5):
    """
    Function to remove outlier data, based on
    the median absolute deviation from the median.
    Note that the function supports asymmetric
    distributions and is based on code from the
    included reference

    Parameters
    ----------
    data: Pandas Series-line
        Series-like containing the simulated
        data in wide format.

    threshold: float
        Threshold of the Median Absolute Deviation
        above which data should be removed

    Returns
    -------
    Array with True values representing index
    of non-outliers

    References
    ----------
    https://eurekastatistics.com/using-the-median-
    absolute-deviation-to-find-outliers/

    TODO:
    -----
    We need to discuss whether this approach is
    appropriate for dropping outlier observations
    """

    m = np.median(data)
    abs_dev = np.abs(data - m)
    left_mad = np.median(abs_dev[data <= m])
    right_mad = np.median(abs_dev[data >= m])
    data_mad = left_mad * np.ones(len(data))
    data_mad[data > m] = right_mad
    z_score = abs_dev / data_mad
    z_score[data == m] = 0
    return z_score < threshold


def is_notebook() -> bool:
    """
    Determine if code is being executed from a Jupyter notebook.
    Taken from https://stackoverflow.com/a/39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
