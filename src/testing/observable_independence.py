# -*- coding: utf-8 -*-
"""
Functions for performing permutation-based, falsification tests of observable,
marginal and conditional independence assumptions.
"""
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm


def _check_array_lengths(
    array_1: np.ndarray, array_2: np.ndarray, array_3: np.ndarray = None
) -> None:
    """
    Ensures that all arrays have equal size.
    """
    size_condition_1 = array_1.size != array_2.size
    size_condition_2 = (
        False if array_3 is None else (array_1.size != array_3.size)
    )
    if size_condition_1 or size_condition_2:
        msg = "All arrays MUST be of the same size."
        raise ValueError(msg)
    return


def _ensure_is_array(array_arg: np.ndarray, name: str) -> None:
    """
    Ensures that `array_arg` is a numpy array array.
    """
    if not isinstance(array_arg, np.ndarray):
        msg = "{} MUST be an instance of np.ndarray.".format(name)
        raise TypeError(msg)
    return


def _create_predictors(array_iterable: Sequence[np.ndarray]) -> np.ndarray:
    """
    Creates the input, 2D numpy array for an sklearn regressor. Each array in
    `array_iterable` is assumed to be 1D.
    """
    if len(array_iterable) > 1:
        combined_predictors = np.concatenate(
            tuple(x[:, None] for x in array_iterable), axis=1
        )
    else:
        combined_predictors = array_iterable[0][:, None]
    return combined_predictors


def _make_regressor(x_2d: np.ndarray, y: np.ndarray) -> LinearRegression:
    """
    Creates a LinearRegression regressor based on the input design matrix
    `x_2d` and target variable `y`.
    """
    # Note, linear regression used instead of Random Forest Regression since
    # the conditional independence test based on RandomForest regression did
    # not pass the unit tests meant to assess whether the test statistic was
    # a u-statistic.
    regressor = LinearRegression()
    regressor.fit(x_2d, y)
    return regressor


def computed_vs_obs_r2(
    x1_array: np.ndarray,
    x2_array: np.ndarray,
    z_array: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    num_permutations: int = 100,
    progress: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Using sklearn's default LinearRegression regressor to predict `x1_array`
    given `x2_array` (and optionally, `z_array`), this function computes
    r2 using the observed `x2_array` and permuted versions of `x2_array`.

    Parameters
    ----------
    x1_array : 1D np.ndarray.
        Denotes the target variable to be predicted.
    x2_array : 1D np.ndarray.
        Denotes the explanatory variable to be used and permuted when trying to
        predict `x1_array`.
    z_array : optional, 1D ndarray or None.
        Detnoes an explanatory variable to be conditioned on, but not to be
        permuted when predicting `x1_array`. Default == None.
    seed : optional, positive int or None.
        Denotes the random seed to be used when permuting `x2_array`.
        Default == None.
    num_permutations : optional, positive int.
        Denotes the number of permutations to use when predicting `x1_array`.
        Default == 100.
    progress : optional, bool.
        Denotes whether or not a tqdm progress bar should be displayed as this
        function is run. Default == True.

    Returns
    -------
    obs_r2 : float
        Denotes the r2 value obtained using `x2_array` to predict `x1_array`,
        given `z_array` if it was not None.
    permuted_r2 : 1D np.ndarray
        Should have length `num_permutations`. Each element denotes the r2
        attained using a permuted version of `x2_array` to predict `x1_array`,
        given `z_array` if it was not None.
    """
    # Validate argument type and lengths
    _ensure_is_array(x1_array, "x1_array")
    _ensure_is_array(x2_array, "x2_array")
    if z_array is not None:
        _ensure_is_array(z_array, "z_array")
    _check_array_lengths(x1_array, x2_array, array_3=z_array)

    # Set a random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Determine how to create the predictors for the permutation test, based
    # on whether we want a marginal independence test (i.e. z_array = None)
    # or a conditional independence test (isinstance(z_array, np.ndarray))
    def create_predictors(array_2):
        if z_array is None:
            return _create_predictors((array_2,))
        return _create_predictors((array_2, z_array))

    # Combine the various predictors
    combined_obs_predictors = create_predictors(x2_array)

    # Determine the number of rows being plotted
    num_rows = x1_array.shape[0]

    # Create a regressor to be used to compute the conditional expectations
    regressor = _make_regressor(combined_obs_predictors, x1_array)

    # Get the observed expectations
    obs_expectation = regressor.predict(combined_obs_predictors)
    obs_r2 = r2_score(x1_array, obs_expectation)

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
        current_x2 = x2_array[shuffled_index_array]
        # Get the current combined predictors
        current_predictors = create_predictors(current_x2)
        # Fit a new model and store the current expectation
        current_regressor = _make_regressor(current_predictors, x1_array)
        permuted_expectations[:, i] = current_regressor.predict(
            current_predictors
        )
        permuted_r2[i] = r2_score(x1_array, permuted_expectations[:, i])
    return obs_r2, permuted_r2


def visualize_permutation_results(
    obs_r2: float,
    permuted_r2: np.ndarray,
    verbose: bool = True,
    permutation_color: str = "#a6bddb",
    output_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
) -> float:
    """
    Parameters
    ----------
    obs_r2 : float
        Denotes the r2 value obtained using `x2_array` to predict `x1_array`,
        given `z_array` if it was not None.
    permuted_r2 : 1D np.ndarray
        Should have length `num_permutations`. Each element denotes the r2
        attained using a permuted version of `x2_array` to predict `x1_array`,
        given `z_array` if it was not None.
    verbose : optional, bool.
        Denotes whether or not the p-value of the permutation test will be
        printed to the stdout. Default == True.
    permutation_color : optional, str.
        Denotes the color of the kernel density estimate used to visuale the
        distribution of r2 from the permuted values of `x2_array`.
        Default == '#a6bddb'.
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
    p_value : float.
        The p-value of the visual permutation test, denoting the percentage of
        times that the r2 with permuted `x2_array` was greater than the r2 with
        the observed `x2_array`.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    p_value = (obs_r2 < permuted_r2).mean()

    if verbose:
        msg = "The p-value of the permutation independence test is {:.2f}."
        print(msg.format(p_value))

    sbn.kdeplot(permuted_r2, ax=ax, color=permutation_color, label="Simulated")

    v_line_label = "Observed\np-val: {:0.3f}".format(  # noqa: F522
        p_value, precision=1
    )
    ax.vlines(
        obs_r2,
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        linestyle="dashed",
        color="black",
        label=v_line_label,
    )

    ax.set_xlabel(r"$r^2$", fontsize=13)
    ax.set_ylabel(
        "Density", fontdict={"fontsize": 13, "rotation": 0}, labelpad=40
    )
    ax.legend(loc="best")
    sbn.despine()

    if output_path is not None:
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig=fig)
    return p_value


def visual_permutation_test(
    x1_array: np.ndarray,
    x2_array: np.ndarray,
    z_array: Optional[np.ndarray] = None,
    num_permutations: int = 100,
    seed: int = 1038,
    progress: bool = True,
    verbose: bool = True,
    permutation_color: str = "#a6bddb",
    output_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
) -> float:
    """
    Performs a visual permutation test of the hypothesis that the expected
    value of `x1_array`, given `x2_array` and (optionally) `z_array`, is equal
    to the expected value of `x1_array` given permuted values of `x2_array` and
    (optionally) `z_array`.

    Parameters
    ----------
    x1_array : 1D np.ndarray.
        Denotes the target variable to be predicted.
    x2_array : 1D np.ndarray.
        Denotes the explanatory variable to be used and permuted when trying to
        predict `x1_array`.
    z_array : optional, 1D ndarray or None.
        Detnoes an explanatory variable to be conditioned on, but not to be
        permuted when predicting `x1_array`. Default == None.
    num_permutations : optional, positive int.
        Denotes the number of permutations to use when predicting `x1_array`.
        Default == 100.
    seed : optional, positive int or None.
        Denotes the random seed to be used when permuting `x2_array`.
        Default == None.
    progress : optional, bool.
        Denotes whether or not a tqdm progress bar should be displayed as this
        function is run. Default == True.
    verbose : optional, bool.
        Denotes whether or not the p-value of the permutation test will be
        printed to the stdout. Default == True.
    permutation_color : optional, str.
        Denotes the color of the kernel density estimate used to visuale the
        distribution of r2 from the permuted values of `x2_array`.
        Default == '#a6bddb'.
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
    p_value : float.
        The p-value of the visual permutation test, denoting the percentage of
        times that the r2 with permuted `x2_array` was greater than the r2 with
        the observed `x2_array`.
    """
    # Compute the observed r2 and the permuted r2's
    obs_r2, permuted_r2 = computed_vs_obs_r2(
        x1_array,
        x2_array,
        z_array=z_array,
        seed=seed,
        num_permutations=num_permutations,
        progress=progress,
    )

    # Visualize the results of the permutation test
    p_value = visualize_permutation_results(
        obs_r2,
        permuted_r2,
        verbose=verbose,
        permutation_color=permutation_color,
        output_path=output_path,
        show=show,
        close=close,
    )
    return p_value
