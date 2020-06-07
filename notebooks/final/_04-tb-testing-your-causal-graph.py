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

# # Testing your causal graph
#
# ## Purpose

# ## Set notebook parameters

# +
# Declare paths to data
DATA_PATH =\
    '../../data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv'

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
# -

# ## Import needed libraries

# +
# Built-in modules
import sys

# Third party modules
import pandas as pd

# Local modules
sys.path.insert(0, '../../')
import src.viz.sim_cdf as sim_cdf
import src.testing.observable_independence as oi
import src.testing.latent_independence as li

from src.graphs.drive_alone_utility import DRIVE_ALONE_UTILITY
from src.utils import sample_from_factor_model
# -

# ## Load and describe needed data

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
# -

# ## Show the posited causal graph

# Draw the causal model being tested
causal_graph = DRIVE_ALONE_UTILITY.draw()
causal_graph.graph_attr.update(size="10,6")
causal_graph

# ## Marginal independence tests
#
# ### Main idea



# ## Conditional independence tests
#
# ### Main idea



# ### Caveats and pitfalls



# ## Latent conditional independence tests
#

# ### Show the posited causal graph



# ### Main idea



# ### Caveats and pitfalls



# ## Summary
