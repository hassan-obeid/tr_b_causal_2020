"""
Functions used to simulate the availability
of alternatives based on the observed availability
of alternatives in a long format dataset.
"""
import random

import numpy as np
import pandas as pd

# Functions to replace within SimulateAvailability

# Function to record number of available alternatives
# in each observation


def get_num_of_av_alts(data_long, obs_id_col):
    """
    Get the number of available alternatives
    from a long format dataset.
    """
    # Create Empty Series to store num of
    # available alternatives
    series = pd.Series([])
    # Get the unique observations
    observations = data_long[obs_id_col]
    unique_observations = observations.unique()
    index_observations = np.arange(len(unique_observations))
    # loop around unique observations and populate
    # the empty series
    for i, obs in zip(index_observations, unique_observations):
        series[i] = data_long[observations == obs].shape[0]
    return series


# Function to simulate availability matrix
def sim_availability_matrix(num_alts, sim_size, alt_name_dict):
    """
    Get the availability matrix based on the number of
    available alternatives, the simulation size,
    and the alternative name dictionary
    """
    # Simulate number of available alternatives for
    # each observation in sim_data
    av_size = sim_size
    alts_sim = np.random.choice(
        a=np.arange(num_alts.max() + 1),
        p=np.bincount(num_alts) / len(num_alts),
        size=av_size,
    )

    # simulate the availability matrix based on number
    # of available alternatives
    N = len(alt_name_dict)
    av_sim = [np.array([1] * K + [0] * (N - K)) for K in alts_sim]

    # Shuffle the available alternatives for each observation
    # because av_sim will always start with 1s
    for x in av_sim:
        np.random.shuffle(x)

    # Shuffle the availability across different observations
    np.random.shuffle(av_sim)

    # Create columns for the availability matrix
    AV_columns = [alt_name_dict[i] + "_AV" for i in alt_name_dict.keys()]

    # Create alternative availability matrix with AV_columns
    AV_Df = pd.DataFrame(data=av_sim, columns=AV_columns)
    return AV_Df


def simulate_availability(data_long, obs_id_col, alt_name_dict, sim_size=1000):
    """
    Function to simulate alternative availability based on a long format
    dataset and join the availability data to the simulated dataset
    resulting from SimDf.

    Parameters
    ----------
    data_long: Pandas DataFrame
        Long format dataframe used for simulating
        alternative availability.

    sim_size: int
        Size of the simulated dataset

    obs_id_col: string
        Name of the column in data_long with
        observation ids.

    alt_name_dic: dictionary
        Dictionary with keys as the ordered number
        of alternatives, and the value for each key
        is a string representing the name of the
        alternative.

    Returns
    -------
    Wide format Pandas DataFrame with additional availability
    columns for each of the alternatives.

    """
    # Get an array of the number of available alternatives
    num_alts = get_num_of_av_alts(data_long, obs_id_col)

    # Create an availability dataframe
    AV_Df = sim_availability_matrix(num_alts, sim_size, alt_name_dict)

    return AV_Df


# Function to generate fake choice column
# this functionality will be relocated to
# a different function, most likely a function
# that merges availability matrix and simulated
# data or a function that converts to long_data


def sim_fake_choice_col(AV_matrix):
    """
    Simulates fake choice column
    needed in the simulated wide format
    dataset.
    """
    # Create an random choice column based on available
    # alternatives for each observation - This column will
    # be needed when converting to long data -- this will
    # be moved to a different column
    available_alt_indices = lambda av_row: np.nonzero(av_row == 1)[0]
    fake_choice = [
        random.choice(available_alt_indices(a)) + 1
        for a in np.array(AV_matrix)
    ]
    return fake_choice
