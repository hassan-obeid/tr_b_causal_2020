# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Declare paths to data
DATA_PATH =\
    '../../data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv'
# Note that these files are based on using the PPCA model
# of Wang and Blei (2018). W represents global factor
# coefficients and Z represents latent factor loadings
PATH_TO_W_PARAMS = '../../data/processed/W_inferred_PPCA.csv'
PATH_TO_Z_PARAMS = '../../data/processed/Z_inferred_PPCA.csv'

# Note the columns of interest for this notebook
MODE_ID_COLUMN = 'mode_id'
OBS_ID_COLUMN = 'observation_id'

TIME_COLUMN = 'total_travel_time'
COST_COLUMN = 'total_travel_cost'
DISTANCE_COLUMN = 'total_travel_distance'
LICENSE_COLUMN = 'num_licensed_drivers'
NUM_AUTOS_COLUMN = 'num_cars'

UTILITY_COLUMNS =\
    [TIME_COLUMN,
     COST_COLUMN,
     DISTANCE_COLUMN,
     LICENSE_COLUMN,
     NUM_AUTOS_COLUMN]

# Note the travel mode of intersest for this notebook
DRIVE_ALONE_ID = 1

# Note the number of permutations to be used when
# testing the causal graphs
NUM_PERMUTATIONS = 100

# Choose a color to represent reference /
# permutation-based test statistics
PERMUTED_COLOR = '#a6bddb'

# +
# Built-in modules
import sys
import itertools

# Third party modules
import numpy as np
import pandas as pd
import scipy.stats

# Local modules
sys.path.insert(0, '../../')
import src.viz.sim_cdf as sim_cdf
import src.testing.observable_independence as oi
import src.testing.latent_independence as li

from src.graphs.drive_alone_utility import (DRIVE_ALONE_UTILITY)
from src.utils import sample_from_factor_model

# +
# Load the raw data
df = pd.read_csv(DATA_PATH)

# Look at the data being used in this notebook
print(df.loc[df[MODE_ID_COLUMN] == DRIVE_ALONE_ID,
             UTILITY_COLUMNS + [OBS_ID_COLUMN]]
        .head(5)
        .T)

# Create a dataframe with the variables posited
# to make up the drive-alone utility
drive_alone_df =\
    df.loc[df[MODE_ID_COLUMN] == DRIVE_ALONE_ID,
           UTILITY_COLUMNS]

# Figure out how many observations we have with
# the drive alone mode being available
num_drive_alone_obs = drive_alone_df.shape[0]

# -

# # PC Algorithm

# ## Step 1: construct the fully connected graph



# ## Step 2 : Test all "0-order interactions," i.e., marginal independencies

# +
# Get all pairs of variables
col_pairs = list(itertools.combinations(UTILITY_COLUMNS, 2))

# Set a seed for reproducbility
np.random.seed(938)

# Test the marginal independencies of all pairs of variables
for col_1, col_2 in col_pairs:
    col_1_array = drive_alone_df[col_1].values
    col_2_array = drive_alone_df[col_2].values
    
    print('{} vs {}:'.format(col_1, col_2))
    oi.visual_permutation_test(
        col_1_array,
        col_2_array,
        z_array=None,
        num_permutations=NUM_PERMUTATIONS,
        permutation_color=PERMUTED_COLOR)
# -

# From the results above, the joint distributions of the following variable pairs merit further investigation for marginal independence:
# - total_travel_distance vs num_licensed_drivers
# - total_travel_cost vs num_licensed_drivers
# - total_travel_time vs num_licensed_drivers
#
# I should take the following investigatory actions:
# 1. Look at the data, i.e., the bivariate plots of the pairs of variables.
# 2. Examine the data summaries, i.e. the models, being used for under-fitting and sensibility (e.g. should this be a linear regression at all?).
# 3. Check whether the suggested independencies (alone and as a collection) make sense logically.
# 4. Try a predictive independence test based on predicting the variance of the conditional distributions instead of predicting the mean.
#
# For now though, I'm going to keep going tonight (June 15th, 2020) in an attempt to learn as many new things as possible, instead of doing the most thorough job I can on each substep.
# I will come back to the actions listed above.

# ## Step 3: Update the working graph

# +
# Remove the edges given by the pairs of variables that passed the conditional independence tests.
# -

# ## Step 4: Test for all "1st-order" interactions, i.e., conditional independences

triplets = list(itertools.permutations(UTILITY_COLUMNS, 3))


# Test the conditional independencies of all triplets of variables
for col_1, col_2, col_3 in triplets:
    col_1_array = drive_alone_df[col_1].values
    col_2_array = drive_alone_df[col_2].values
    col_3_array = drive_alone_df[col_3].values
    
    print('{} indep {} given {}:'.format(col_1, col_2, col_3))
    oi.visual_permutation_test(
        col_1_array,
        col_2_array,
        z_array=col_3_array,
        num_permutations=NUM_PERMUTATIONS,
        permutation_color=PERMUTED_COLOR)
