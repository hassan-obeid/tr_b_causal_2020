# -*- coding: utf-8 -*-
"""
Functions for performing permutation-based conditional independence tests.
"""
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import seaborn as sbn
import matplotlib.pyplot as plt

from tqdm import tqdm


def _make_regressor(x_2d, y):
    # Note, linear regression used instead of Random Forest Regression since
    # the conditional independence test based on RandomForest regression did
    # not pass the unit tests meant to assess whether the test statistic was
    # a u-statistic.
    regressor = LinearRegression()
    regressor.fit(x_2d, y)
    return regressor


def computed_vs_obs_r2(df,
                       x1_col,
                       x2_col,
                       z_col,
                       mode_id_col,
                       seed,
                       num_permutations=100,
                       progress=True):
    # Set a random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Get the values to be used for testing
    filter_array = (df[mode_id_col] == 1).values
    obs_x1 = df[x1_col].values[filter_array]
    obs_x2 = df[x2_col].values[filter_array]
    obs_z = df[z_col].values[filter_array]

    # Combine the various predictors
    combined_obs_predictors =\
        np.concatenate((obs_x2[:, None], obs_z[:, None]), axis=1)

    # Determine the number of rows being plotted
    num_rows = obs_x1.shape[0]

    # Create a regressor to be used to compute the conditional expectations
    regressor = _make_regressor(combined_obs_predictors, obs_x1)

    # Get the observed expectations
    obs_expectation = regressor.predict(combined_obs_predictors)
    obs_r2 = r2_score(obs_x1, obs_expectation)

    # Initialize arrays to store the permuted expectations and r2's
    permuted_expectations = np.empty((num_rows, num_permutations))
    permuted_r2 = np.empty(num_permutations, dtype=float)

    # Get the permuted expectations
    shuffled_index_array = np.arange(num_rows)

    iterable = range(num_permutations)
    if progress:
        iterable = tqdm(iterable)

    for i in iterable:
        # Shuffle the index array
        np.random.shuffle(shuffled_index_array)
        # Get the new set of permuted X_2 values
        current_x2 = obs_x2[shuffled_index_array]
        # Get the current combined predictors
        current_predictors =\
            np.concatenate((current_x2[:, None], obs_z[:, None]), axis=1)
        # Fit a new model and store the current expectation
        current_regressor =\
            _make_regressor(current_predictors, obs_x1)
        permuted_expectations[:, i] =\
            current_regressor.predict(current_predictors)
        permuted_r2[i] = r2_score(obs_x1, permuted_expectations[:, i])
    return obs_r2, permuted_r2


def visualize_permutation_results(obs_r2,
                                  permuted_r2,
                                  verbose=True,
                                  permutation_color='#a6bddb',
                                  show=True,
                                  close=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    p_value = (obs_r2 < permuted_r2).mean()

    if verbose:
        msg =\
            'The p-value of the permutation C.I.T. is {:.2f}.'.format(p_value)
        print(msg)

    sbn.kdeplot(
        permuted_r2, ax=ax, color=permutation_color, label='Simulated')
    ax.vlines(obs_r2,
              ax.get_ylim()[0],
              ax.get_ylim()[1],
              linestyle='dashed',
              color='black',
              label='Observed\np-val: {:0.3f}'.format(p_value,
                                                      precision=1))

    ax.set_xlabel(r'$r^2$', fontsize=13)
    ax.set_ylabel(
        'Density', fontdict={'fontsize':13, 'rotation':0}, labelpad=40)
    ax.legend(loc='best')
    sbn.despine()
    if show:
        plt.show()
    if close:
        plt.close(fig=fig)
    return p_value


def visual_permutation_test(df,
                            x1_col,
                            x2_col,
                            z_col,
                            mode_id_col,
                            num_permutations=100,
                            seed=1038,
                            progress=True,
                            verbose=True,
                            permutation_color='#a6bddb',
                            show=True,
                            close=False):
    # Compute the observed r2 and the permuted r2's
    obs_r2, permuted_r2 =\
        computed_vs_obs_r2(
            df, x1_col, x2_col, z_col,
            mode_id_col, seed,
            num_permutations=num_permutations, progress=progress)

    # Visualize the results of the permutation test
    p_value =\
        visualize_permutation_results(
            obs_r2,
            permuted_r2,
            verbose=verbose,
            permutation_color=permutation_color,
            show=show,
            close=close)
    return p_value
