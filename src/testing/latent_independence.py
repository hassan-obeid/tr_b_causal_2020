# -*- coding: utf-8 -*-
"""
Functions for performing permutation-based, falsification tests of latent,
marginal and conditional independence assumptions.
"""
# Built-in modules
from numbers import Number

# Third-party modules
import numpy as np

import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Union, Tuple, Optional

# Local modules
import src.viz as viz
import src.viz.sim_cdf as sim_cdf
import src.testing.observable_independence as oi


def compute_pvalue(obs_statistic: Union[Number, np.ndarray],
                   reference_statistics: np.ndarray) -> float:
    """
    Computes the percentage of time that `obs_statistic` is
    less than `reference_statistics`.

    Parameters
    ----------
    obs_statistic : Number or 1D ndarray.
        Denotes the values of whatever test statistic is
        being calculated, based on observed data.
    reference_statistics : 1D ndarray.
        Denotes the values of the test statistic, based on
        one's reference data (e.g. permutated data, simulated
        data, etc.)

    Returns
    -------
    p_value : float.
        The percentage of times `obs_statistic` is less than
        `reference_statistics`.
    """
    return (obs_statistic < reference_statistics).mean()


def compute_predictive_independence_test_values(
        samples: np.ndarray,
        obs_sample: np.ndarray,
        seed: int=1038,
        num_permutations: int=100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes two arrays of p-values, one using only simulated data
    from `samples` and the other using observed data from `obs_sample`.
    Each p-value in the arrays is the result of a conditional independence
    test based on a simulated vector of a conditioning variable.

    Parameters
    ----------
    samples : 3D ndarray.
        Should have shape (num_observations, 3, num_samples). The three
        columns should represent X1, X2, and Z, for a test of X1
        independent of X2 conditional on Z. All values should be sampled
        from a single joint distribution that satisfies the conditional
        independence assumptions.
    obs_sample : 2D ndarray
        Should have shape (num_observations, 2). The two columns should
        represnt X1 and x2 for the conditional independence test.
    seed : optional, positive int.
        The random seed to be used to ensure reproducibility.
        Default == 1038.
    num_permutations : optional, positive int.
        Denotes the number of permutations to be used in the independence
        test. Default == 100.

    Returns
    -------
    sampled_pvals, obs_pvals : 1D np.ndarray
        Denotes the p-values calculated from a conditional mean independence
        test, using (respectively) sampled / simulated values of X1, X2, and
        Z or using observed X1, X2 and simulated Z.
    """
    # Determine the number of samples in order to create an iterable for
    # getting and storing test samples
    if len(samples.shape) != 3:
        msg = '`samples` should have shape (num_rows, 3, num_samples).'
        raise ValueError(msg)
    num_samples = samples.shape[-1]

    # Initialize a container for the p-values of the sampled and observed data
    sampled_pvals = np.empty((num_samples,), dtype=float)
    obs_pvals = np.empty((num_samples,), dtype=float)

    # Create the iterable to be looped over to compute test values
    iterable = viz.progress(range(num_permutations))

    # Populate the arrays of test statistics
    for i in iterable:
        # Get the data to be used to calculate this set of p-values
        current_sim_sample = samples[:, :, i]
        current_sim_z = current_sim_sample[:, -1]
        current_augmented_obs =\
            np.concatenate((obs_sample, current_sim_z[:, None]),
                           axis=1)
        current_seed = seed + i

        # Package the arguments to compute the predictive r2 values
        sim_args =\
            (current_sim_sample[:, 0],
             current_sim_sample[:, 1],
             current_sim_z,
             current_seed,
             num_permutations,
             False
            )

        augmented_obs_args =\
            (current_augmented_obs[:, 0],
             current_augmented_obs[:, 1],
             current_augmented_obs[:, 2],
             current_seed,
             num_permutations,
             False
            )

        # Compute and store the p-values of the conditional independence
        # test for the current simulated and augmented dataset
        sampled_pvals[i] =\
            compute_pvalue(*oi.computed_vs_obs_r2(*sim_args))

        obs_pvals[i] =\
            compute_pvalue(*oi.computed_vs_obs_r2(*augmented_obs_args))
    return sampled_pvals, obs_pvals


def visualize_predictive_cit_results(
        sampled_pvals: np.ndarray,
        obs_pvals: np.ndarray,
        simulated_color: str='#a6bddb',
        observed_color: str='#045a8d',
        verbose: bool=True,
        output_path: Optional[str]=None,
        show: bool=True,
        close: bool=False) -> float:
    """
    Plots the distributions of simulation-based and observation-based
    p-values, displaying the results of the latent, conditional
    mean-independence test.

    Parameters
    ----------
    sampled_pvals, obs_pvals : 1D np.ndarray
        Denotes the p-values calculated from a conditional mean independence
        test, using (respectively) sampled / simulated values of X1, X2, and
        Z or using observed X1, X2 and simulated Z.
    verbose : optional, bool.
        Denotes whether or not the p-value of the permutation test will be
        printed to the stdout. Default == True.
    output_path : optional, str or None.
        Denotes the path to the location where the plot visualizing the
        permutation test results will be stored. If `output_path` is None, the
        plot will not be stored. Default is None.
    show : optional, bool.
        Denotes whether the matplotlib figure that visualizes the results of
        the permutation test should be shown. Default == True.
    close : optional, bool.
        Denotes whether the matplotlib figure that visualizes the results of
        the permutation test should be closed. Default == False.

    Returns
    -------
    overall_pvalue : float
        The overall p-value for the latent, conditional mean independence test.
    """
    sbn.set_style('white')
    fig, ax = plt.subplots(figsize=(10, 6))
    overall_p_value = (obs_pvals < sampled_pvals).mean()

    if verbose:
        msg =\
            'The p-value of the predictive, permutation C.I.T. is {:.2f}.'
        print(msg.format(overall_p_value))

    sbn.distplot(
        sampled_pvals, ax=ax, color=simulated_color, label='Simulated', kde=False)
    sbn.distplot(
        obs_pvals, ax=ax, color=observed_color, label='Observed', kde=False)

    ax.set_xlim(left=0)

    ax.set_xlabel('Permutation P-value', fontsize=13)
    ax.set_ylabel(
        'Density', fontdict={'fontsize':13, 'rotation':0}, labelpad=40)
    ax.legend(loc='best')
    sbn.despine()

    if output_path is not None:
        fig.savefig(output_path, dpi=500, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close(fig=fig)
    return overall_p_value


def perform_visual_predictive_cit_test(
        samples: np.ndarray,
        obs_sample: np.ndarray,
        seed: int=1038,
        num_permutations: int=100,
        verbose: bool=True,
        output_path: Optional[str]=None,
        show: bool=True,
        close: bool=False) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Performs a visual, permutation test of the hypothesis that the expected
    value of `x1_array`, given `x2_array` and a latent `z_array`, is equal
    to the expected value of `x1_array` given permuted values of `x2_array`
    and the latent `z_array`.

    Parameters
    ----------
    samples : 3D ndarray of shape (num_rows, 3, num_samples).
        Columns should contain, in order, simulated x1, x2, z.
    obs_sample : 2D ndarray of shape (num_rows, 2)
        Columns should contain, in order, observed x1, observed x2.
    seed : optional, positive int.
        The random seed to be used to ensure reproducibility.
        Default == 1038.
    num_permutations : optional, positive int.
        Denotes the number of permutations to be used in the independence
        test. Default == 100.
    verbose : optional, bool.
        Denotes whether or not the p-value of the permutation test will be
        printed to the stdout. Default == True.
    output_path : optional, str or None.
        Denotes the path to the location where the plot visualizing the
        permutation test results will be stored. If `output_path` is None, the
        plot will not be stored. Default is None.
    show : optional, bool.
        Denotes whether the matplotlib figure that visualizes the results of
        the permutation test should be shown. Default == True.
    close : optional, bool.
        Denotes whether the matplotlib figure that visualizes the results of
        the permutation test should be closed. Default == False.

    Returns
    -------
    overall_pvalue : float
        The overall p-value for the latent, conditional mean independence test.
    sampled_pvals, obs_pvals : 1D np.ndarray
        Denotes the p-values calculated from a conditional mean independence
        test, using (respectively) sampled / simulated values of X1, X2, and
        Z or using observed X1, X2 and simulated Z.
    """
    # Compute the observed and sampled pvalues
    sampled_pvals, obs_pvals =\
        compute_predictive_independence_test_values(
            samples, obs_sample, seed, num_permutations=num_permutations)

    # Visualize the results of the predictive permutation CIT test
    overall_p_value =\
        visualize_predictive_cit_results(
            sampled_pvals,
            obs_pvals,
            verbose=verbose,
            output_path=output_path,
            show=show, close=close)
    return overall_p_value, sampled_pvals, obs_pvals


def plot_simulated_vs_observed_cdf(
    obs_array: np.ndarray,
    simulated_array: np.ndarray,
    obs_color: str='#045a8d',
    sim_color: str='#a6bddb',
    figsize: Tuple[int, int]=(10, 6),
    x_label: str='X',
    output_path: Optional[str]=None,
    show: bool=True,
    close: bool=False) -> Axes:
    """
    Plots both simulated and observed cdfs for provided observed and simulated
    data arrays. Useful for sanity checking whether the marginal distribution
    of the observed data resembles the marginal distribution of the simulated
    data.

    Parameters
    ----------
    obs_array : 1D ndarray.
        Denotes the array of observed data whose CDF is to be plotted.
    simulated_array : 2D ndarray.
        Denotes the array of simulated data whose CDF is to be plotted. Each
        column is assumed to represent a different simulated vector of values.
    obs_color, sim_color : optional, str.
        Denotes the matplotlib color codes for the observed and simulated CDFs,
        respectively. Default colors are '#045a8d' and '#a6bddb', respectively.
    figsize : optional, 2-tuple of ints.
        Denotes the width and height, in inches, of the resulting figure.
    x_label : optional, str.
        The label to be displayed on the x-axis. Default == 'X'
    output_path : optional, str or None.
        Denotes the path to the location where the plot visualizing the
        permutation test results will be stored. If `output_path` is None, the
        plot will not be stored. Default is None.
    show : optional, bool.
        Denotes whether the matplotlib figure that visualizes the results of
        the permutation test should be shown. Default == True.
    close : optional, bool.
        Denotes whether the matplotlib figure that visualizes the results of
        the permutation test should be closed. Default == False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes instance containing the simulated and observed CDFs.
    """
    sbn.set_style('white')
    # Determine the number of simulations
    num_simulations = simulated_array.shape[1]

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the simulated cdfs
    for sim_id in range(num_simulations):
        label = 'Simulated' if sim_id == 0 else None
        sim_cdf._plot_single_cdf_on_axis(
            simulated_array[:, sim_id],
            ax,
            color=sim_color,
            label=label, alpha=0.5)

    # Plot the observed cdf
    sim_cdf._plot_single_cdf_on_axis(
        obs_array,
        ax,
        color=obs_color,
        label='Observed',
        alpha=1.0)

    # Label the plot axes
    ax.set_xlabel("{}".format(x_label), fontsize=12)
    ax.set_ylabel("Cumulative\nDensity\nFunction",
                  rotation=0,
                  labelpad=40,
                  fontsize=12)

    # Show the line labels
    plt.legend(loc='best')

    sbn.despine()
    if output_path is not None:
        fig.savefig(output_path, dpi=500, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close(fig=fig)
    return ax
