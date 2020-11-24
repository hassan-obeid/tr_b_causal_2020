"""
Functions used in the fitting of distributions
to variables to be simulated.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from fitter import Fitter

# Functions to replace code within
# DistNodeNoParent
# Function for checking length


def is_unique(var_values):
    """
    Checks whether a variable has one unique value.
    """
    return len(var_values.unique()) == 1


# def is_empirical(var_type):
#     """
#     Checks whether the variable type for the
#     variable of interest is to be taken
#     as a constant value or as numerical values.
#     """
#     return var_type in ['constant', 'numerical']


def is_constant(var_type):  # to be rethought
    """
    Checks whether a variable has a constant
    value.
    """
    return var_type == "constant"


def is_empirical(var_type):
    """
    Checks whether the variable type for the
    variable of interest is to be taken
    as a constant value or as numerical values.
    """
    return var_type == "empirical"


def is_categorical(var_type):
    """
    Checks whether the variable type for the
    variable of interest is categorical.
    """
    return var_type == "categorical"


def get_alt_specific_variable_name(var_name, alt_name):
    """
    Gets the alternative specific variable,
    returns a string starting with variable name and
    ending with alternative name.
    """
    return var_name + "_" + alt_name


def get_constant_dist(var, var_val, alt_name=None):
    """
    Retrives the constant 'distribution' of a
    constant variable.
    """
    constant_dict = defaultdict(dict)
    # Add name of alternative to variable and store distriburion & parameters
    var_name = (
        var
        if alt_name is None
        else get_alt_specific_variable_name(var, alt_name)
    )
    constant_dict[var_name]["distribution"] = "constant"
    constant_dict[var_name]["parameters"] = var_val.unique()
    return constant_dict


def get_empirical_dist(var, var_val, alt_name=None):
    """
    Retrives the empirical values of the alternative
    specific variable of interest as its distribution.
    """
    empir_dict = defaultdict(dict)
    # Add name of alternative to variable and store distriburion & parameters
    var_name = (
        var
        if alt_name is None
        else get_alt_specific_variable_name(var, alt_name)
    )
    empir_dict[var_name]["distribution"] = "empirical"
    empir_dict[var_name]["parameters"] = np.array(var_val)
    return empir_dict


def get_categorical_dist(var, var_val, alt_name=None):
    """
    Retrives the unique values and the proportions
    of observed values for a categorical alternative
    specific variables.
    """
    categ_dict = defaultdict(dict)
    # If more than one category, compute the frequency of values
    # and store as parameters
    # Add name of alternative to variable and store distriburion & parameters
    var_name = (
        var
        if alt_name is None
        else get_alt_specific_variable_name(var, alt_name)
    )
    categ_dict[var_name]["distribution"] = "categorical"
    np_array_range = np.arange(var_val.max() + 1)
    array_bincount = np.bincount(var_val)
    probs = array_bincount / len(var_val)
    categ_dict[var_name]["parameters"] = [np_array_range, probs]
    return categ_dict


def get_continuous_dist(var, var_val, cont_dists, alt_name=None):
    """
    Retrives the distribution of continuous alternative
    specific variables using the Fitter package.
    """
    cont_dict = defaultdict(dict)
    # Use the Fitter library to fit distributions
    # to the data
    fitter_object = Fitter(
        data=var_val, distributions=cont_dists, timeout=60
    )
    fitter_object.fit()
    # Get the best distribution and store in dictionary
    BestDict = fitter_object.get_best()
    # Add name of alternative to variable and store distriburion & parameters
    var_name = (
        var
        if alt_name is None
        else get_alt_specific_variable_name(var, alt_name)
    )
    cont_dict[var_name]["distribution"] = list(BestDict.items())[0][0]
    cont_dict[var_name]["parameters"] = list(BestDict.items())[0][1]
    return cont_dict


def get_distribution_dicts(var, var_type, var_val, cont_dists, alt_name=None):
    """
    Helper function to generate a distribution dictionary
    for the variable specified.
    """
    # If data is categorical
    if is_empirical(var_type):
        # If only one category
        if is_unique(var_val):
            # Add name of alternative to variable
            # and store distriburion & parameters
            dist_dic = get_constant_dist(var, var_val, alt_name)
        else:
            dist_dic = get_empirical_dist(var, var_val, alt_name)
    elif is_categorical(var_type):
        if is_unique(var_val):
            dist_dic = get_constant_dist(var, var_val, alt_name)
        else:
            dist_dic = get_categorical_dist(var, var_val, alt_name)
    else:
        # If data is not categorical but has one unique value
        if is_unique(var_val):
            dist_dic = get_constant_dist(var, var_val, alt_name)
        # If data is not categorical but has more than one unique value
        else:
            dist_dic = get_continuous_dist(var, var_val, alt_name, cont_dists)
    return dist_dic


############################################
# Functions to replace functionality for
# fitting distributions for variables
# specific variables that have no parents
# in the causal graph.
############################################


def ind_spec_dist(data_long, obs_id_col, ind_spec, var_types, cont_dists):
    """
    Function that retrieves distributions for all individual
    specific variables.
    """
    ind_spec_dict = defaultdict(dict)
    for ind_var in ind_spec:
        # generate array of values for individual specific variable
        var_val = (
            data_long[[obs_id_col, ind_var]]
            .drop_duplicates(obs_id_col, inplace=False)
            .loc[:, ind_var]
            .reset_index(drop=True)
        )
        # Get distribution of variable
        var_type = var_types[ind_var]
        ind_var_dic = get_distribution_dicts(
            ind_var, var_type, var_val, cont_dists
        )
        ind_spec_dict.update(ind_var_dic)
    return ind_spec_dict


def alt_spec_dist(
    data_long, alt_id_col, alt_spec_dic, var_types, alt_name_dic, cont_dists
):
    """
    Function that retrieves distributions for all alternative
    specific variables.
    """
    all_alt_spec_var_dic = defaultdict(dict)
    for alt_id in data_long[alt_id_col].unique():
        # Store data for specific alternative (mode)
        alt_data = data_long.loc[data_long[alt_id_col] == alt_id]
        alt_spec_var_dic = defaultdict(dict)
        # Loop around the alternative specific
        # variables in the input dictionary
        alt_name = alt_name_dic[alt_id]
        for alt_var in alt_spec_dic[alt_id]:
            var_val = alt_data[alt_var]
            var_type = var_types[alt_var]
            alt_spec_var_dist = get_distribution_dicts(
                alt_var, var_type, var_val, alt_name, cont_dists
            )
            alt_spec_var_dic.update(alt_spec_var_dist)
        all_alt_spec_var_dic.update(alt_spec_var_dic)
    return all_alt_spec_var_dic


def trip_spec_dist(data_long, obs_id_col, trip_spec, var_types, cont_dists):
    """
    Function that retrieves distributions for all trip
    specific variables.
    """
    # Trip Specific Variable (maybe combine with individual specific variables)
    # Loop around trip (observation) specific variables
    trip_spec_dict = defaultdict(dict)
    for trip_var in trip_spec:
        # generate array of values for trip specific variable
        var_val = (
            data_long[[obs_id_col, trip_var]]
            .drop_duplicates(obs_id_col, inplace=False)
            .loc[:, trip_var]
            .reset_index(drop=True)
        )
        var_type = var_types[trip_var]
        # If data is to be taken as empirical values
        trip_spec_var_dist = get_distribution_dicts(
            trip_var, var_type, var_val, cont_dists
        )
        trip_spec_dict.update(trip_spec_var_dist)
    return trip_spec_dict


# Define the main function
def get_dist_node_no_parent(
    data_long,
    alt_id_col,
    obs_id_col,
    alt_spec_dic,
    alt_name_dic,
    ind_spec,
    trip_spec,
    var_types,
    cont_dists=None,
):
    """
    Function to find the distribution of specific variables
    from a long format dataset.

    Parameters
    ----------
    data_long: Pandas DataFrame
        Dataset in long format from which variable
        distribution is to be found.

    alt_id_col: string
        Name of the column with alternative ids.

    obs_id_col: string
        Name of the column with observation ids.

    alt_spec_dic: dictionary
        Dictionary with keys as the ordered number
        of alternatives, and the value for each key
        is a list of strings representing the name of
        variables without parents per alternative.

    alt_name_dic: dictionary
        Dictionary with keys as the ordered number
        of alternatives, and the value for each key
        is a string representing the name of the
        alternative.

    ind_spec: list
        List containing strings of the names of
        individual specific variables.

    trip_spec: list
        List containing string of the names of
        trip specific variables.

    var_types: dictionary
        Dictionary with keys as strings of names of variables
        from long format dataset, and values for each key are
        the type of variables (e.g.: 'categorical vs. continuous').

    cont_dists: list
        List of continuous RVs distribution names from scipy.

    Returns
    -------
    a nested dictionary with keys as variable names and values
    as dictionaries containing both the distribution name and
    its parameters.
    """
    params_dict = defaultdict(dict)

    # Code for Individual Specific Variables
    print("Getting Distributions of Individual Specific Variables...")
    print("---------------------------------------------------------")
    ind_spec_dic_params = ind_spec_dist(
        data_long, obs_id_col, ind_spec, var_types, cont_dists
    )
    params_dict.update(ind_spec_dic_params)
    print("Done...")

    # Code for Alternative Specific Variables
    # Loop around the different available alternatives
    print("Getting Distributions of Alternative Specific Variables...")
    print("----------------------------------------------------------")
    alt_spec_dic_params = alt_spec_dist(
        data_long,
        alt_id_col,
        alt_spec_dic,
        var_types,
        alt_name_dic,
        cont_dists,
    )
    params_dict.update(alt_spec_dic_params)
    print("Done...")

    # Trip Specific Variable (maybe combine with individual specific variables)
    # Loop around trip (observation) specific variables
    print("Getting Distributions of Trip Specific Variables...")
    print("---------------------------------------------------------")
    trip_spec_dic_params = trip_spec_dist(
        data_long, obs_id_col, trip_spec, var_types, cont_dists
    )
    params_dict.update(trip_spec_dic_params)
    print("Done...")

    return params_dict
