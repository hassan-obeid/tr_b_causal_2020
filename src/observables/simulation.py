"""
Functions used in the simulation of observables
to simulate data for nodes in a causal graph
based on specified distribution and parameters
"""


import numpy as np
import pandas as pd
import scipy.stats


def is_unique(var_values):
    """
    Checks whether a variable has one unique value.
    """
    return len(var_values.unique()) == 1


def is_constant(var_type):  # to be rethought
    """
    Checks whether a variable has a constant
    value.
    """
    return var_type == 'constant'


def is_empirical(var_type):
    """
    Checks whether the variable type for the
    variable of interest is to be taken
    as a constant value or as numerical values.
    """
    return var_type == 'empirical'


def is_categorical(var_type):
    """
    Checks whether the variable type for the
    variable of interest is categorical.
    """
    return var_type == 'categorical'


def sim_categorical(var_dist_params, size):
    """
    Function to simulate data for
    a categorical/Discrete variable.
    """
    values = var_dist_params[0]
    freq = var_dist_params[1]
    data_sim = np.random.choice(a=values,
                                p=freq,
                                size=size)
    return data_sim


def sim_constant(var_dist_params):
    """
    Function to simulate data for a
    'constant' variable, in other words,
    a variable that has one empirical value.
    """
    data_sim = var_dist_params
    return data_sim


def sim_empirical(var_dist_params, size):
    """
    Function to sample with replacement
    for a variable.
    """
    data_sim = np.random.choice(var_dist_params, size=size)
    return data_sim


def sim_continuous(var_dist, var_dist_params, size):
    """
    Function to simulate data from a continuous
    distribution.
    """
    # Get scipy distribution from its
    # name in the params dictionary
    dist = getattr(scipy.stats,
                   var_dist)

    data_sim = dist.rvs(*var_dist_params,
                        size=size)
    return data_sim


def sim_from_distribution(var_dist, var_dist_params, size):
    """
    Funtion to simulate data of size N based type of dist
    and the distribution parameters.
    """
    if is_categorical(var_dist):
        sim_array = sim_categorical(var_dist_params, size)
        # Simulate variables for data with a single unique value
    elif is_constant(var_dist):
        sim_array = sim_constant(var_dist_params)
        # Simulate data using values from array, sampling
        # with replacement
    elif is_empirical(var_dist):
        sim_array = sim_empirical(var_dist_params, size)
        # Simulate data for continuous variables
    else:
        sim_array = sim_continuous(var_dist, var_dist_params, size)
    return sim_array


def sim_node_no_parent(params_dict, size=1000):
    """
    Funtion to simulate data of size N based on specified
    distribution/parameters found by the fitter package.

    Paremeters
    ----------
    dist_params: dictionary
        The variable distribution dictionary resulting from
        `FindLongDataDist`.

    size: int
        Size of the desired simulated dataset, default value
        is 1000 observations.

    Returns
    -------
    DataFrame object with simulated data based on specified distributions
    """
    # Create Empty DataFrame with keys from params_dict
    sim_df = pd.DataFrame(columns=list(params_dict.keys()))
    sim_df = sim_df.fillna(0)

    for column in list(params_dict.keys()):
        variable = params_dict[column]
        var_dist = variable['distribution']
        var_dist_params = variable['parameters']
        sim_df[column] = sim_from_distribution(var_dist,
                                               var_dist_params,
                                               size)

    return sim_df