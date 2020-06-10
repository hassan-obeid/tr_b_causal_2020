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

# # To-Do:
# - Fill in overall purpose for the notebook.
# - Fill in caveats and pitfalls for marginal independence tests.
# - Fill in main idea for latent conditional independence tests.
# - Fill in posited causal graph for latent conditional independence tests.
# - Fill in caveats and pitfalls for latent conditional independence tests.
# - Fill in summary for notebook.
#
# # Testing your causal graph
#
# ## Purpose

# ## Set notebook parameters

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
# -

# ## Import needed libraries

# +
# Built-in modules
import sys

# Third party modules
import numpy as np
import pandas as pd
import scipy.stats

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

# Figure out how many observations we have with
# the drive alone mode being available
num_drive_alone_obs = drive_alone_df.shape[0]
# -

# ## Show the posited causal graph

# Draw the causal model being tested
causal_graph = DRIVE_ALONE_UTILITY.draw()
causal_graph.graph_attr.update(size="10,6")
causal_graph

# ## Marginal independence tests
#
# ### Main idea
# The marginal independence tests demonstrated in this notebook will visually test the following implication<br>
# $
# \begin{aligned}
# P \left( X_1 \mid X_2 \right) &= P \left( X_1 \right) \\
# \int x_1 P \left( X_1 \mid X_2 \right) \partial{x_1}  &= \int x_1 P \left( X_1 \right) \partial{x_1} \\
# E \left[ X_1 \mid X_2 \right] &= E \left[ X_1 \right]
# \end{aligned}
# $
#
# In other words, if $X_1$ is marginally independent of $X_2$, then the expectation of $X_1$ conditional on $X_2$ is equal to the marginal expectation of $X_1$.
# Marginal independence implies mean independence.
# This means that shuffling / permuting the $X_2$ columns should make no difference to predicting $X_1$, once one predicts $\bar{X_1}$.
#
# The test demonstrated below works by estimating a linear regression to predict $E \left[ X_1 \mid X_2 \right]$.
# The $r^2$ from these regressions is computed using the observed value of $X_2$, and it is also computed using the permuted values of $X_2$ (which are independent of $X_1$ by construction).
#
# If $X_1$ is marginally independent of $X_2$, then the $r^2$ using the observed values of $X_2$ should be similar to the distribution of $r^2$ using the permuted values of $X_2$.
#
# For this test, we'll use the following marginal independence assumption implied by the causal graph above:<br>
# $
# P \left( \textrm{Number of Automobiles} \mid \textrm{Number of licensed drivers} \right) = P \left( \textrm{Number of Automobiles} \right)
# $

# +
license_array = drive_alone_df[LICENSE_COLUMN].values
num_cars_array = drive_alone_df[NUM_AUTOS_COLUMN].values

oi.visual_permutation_test(
    license_array,
    num_cars_array,
    z_array=None,
    seed=1038,
    num_permutations=NUM_PERMUTATIONS,
    permutation_color=PERMUTED_COLOR)
# -

# ### Caveats and pitfalls
#

# ## Conditional independence tests
#
# ### Main idea
# In particular, the notebook will show one way to visually and numerically test the following implication<br>
# $
# \begin{aligned}
# P \left( X_1 \mid X_2, Z \right) &= P \left( X_1 \mid Z \right) \\
# \int x_1 P \left( X_1 \mid X_2, Z \right) \partial{x_1}  &= \int x_1 P \left( X_1 \mid Z \right) \partial{x_1} \\
# E \left[ X_1 \mid X_2, Z \right] &= E \left[ X_1 \mid Z \right]
# \end{aligned}
# $
#
# In other words, if $X_1$ is conditionally independent of $X_2$ given $Z$, then the expectation of $X_1$ conditional on $X_2$ and $Z$ is equal to the expectation of $X_1$ conditional on just $Z$.
# This implies that shuffling / permuting $X_2$ should make no difference for predicting $X_1$ once we've included $Z$ while predicting.
#
# In other words, one's ability predict to predict $X_1$ should not depend on whether one uses the original $X_2$ or the permuted $X_2$, as long as one conditions on $Z$ when predicting $X_1$.
# This invariance will be tested by using a simple predictive model, linear regression, and comparing $r^2$ as a measure of predictive ability when using $Z$ and the original $X_2$ versus $r^2$ when using $Z$ and the permuted $X_2$.
#
# For this test, we'll use the following conditional independence assumption implied by the causal graph above:<br>
# $
# P \left( \textrm{Travel Time} \mid \textrm{Travel Cost}, \textrm{Travel Distance} \right) = P \left( \textrm{Travel Time} \mid \textrm{Travel Distance} \right)
# $

# +
time_array = drive_alone_df[TIME_COLUMN].values
cost_array = drive_alone_df[COST_COLUMN].values
distance_array = drive_alone_df[DISTANCE_COLUMN].values

oi.visual_permutation_test(
    time_array,
    cost_array,
    z_array=distance_array,
    num_permutations=NUM_PERMUTATIONS,
    permutation_color=PERMUTED_COLOR)
# -

# ### Caveats and pitfalls
# When testing conditional mean independence (i.e., $E \left[ X_1 \mid X_2, Z \right] = E \left[ X_1 \mid Z \right]$), there are four potential issues of interest.
#
# The first issue is misspecification of $E \left[ X_1 \mid Z \right]$.
# If the conditional mean of $X_1$ is affected by un-modeled functions of $Z$ and $Z \rightarrow X_2$, then the inclusion of $X_2$ in one's model may simply act as a proxy for the un-modeled function of $Z$.
# Such proxy behavior would lead one to observe "inflated" values of $r^2$ when modeling $E \left[ X_1 \mid X_2, Z \right]$, thus increasing the probability that one will falsely reject the null-hypothesis of conditional independence.
#
# In other words, to guard against higher-than-nominal probabilities of type-1 error, one needs to guard against underfitting of the model for $E \left[ X_1 \mid Z \right]$ **before** computing one's test-statistic and (permutation-based) reference-distribution.
#
# The second issue is misspecification of $E \left[ X_1 \mid X_2, Z \right]$.
# If one's model for $E \left[ X_1 \mid X_2, Z \right]$ is underfit with respect to $X_2$, then one's test-statistic ($r^2$) will be lower than it should be under accurate specification.
# This leads to lower probability of correctly rejecting the null-hypothesis.
#
# The last two issues are the converse of the problems and effects above.
# If one's model for $E \left[ X_1 \mid Z \right]$ is overfit, then the reference-distribution of the test statistic will be artificially high.
# This will lead to lower probability of correctly rejecting the null hypothesis of conditional independence.
# Likewise, if one's model for $E \left[ X_1 \mid X_2, Z \right]$ is overfit with respect to $X_2$, then one's test-statistic will be artificially high, thus increasing the probability of incorrectly rejecting the null-hypothesis.
#
# Visually, the way to think about these four issues is that, relative to using well-specified models for $E \left[ X_1 \mid Z \right]$ and $E \left[ X_1 \mid X_2, Z \right]$:
# - underfitting of $E \left[ X_1 \mid Z \right]$ shifts one's reference distribution to the left, while leaving one's observed test statistic unchanged;
# - underfitting of $E \left[ X_1 \mid X_2, Z \right]$ shifts one's observed test statistic to the left, while leaving one's reference distribution unchanged;
# - overfitting of $E \left[ X_1 \mid Z \right]$ shifts one's reference distribution to the right, while leaving one's observed test statistic unchanged;
# - overfitting of $E \left[ X_1 \mid X_2, Z \right]$ shifts one's observed test statistic to the right, while leaving one's reference distribution unchanged.
#
# To avoid all such problems or the combination of these problems, be sure to check one's models of $E \left[ X_1 \mid Z \right]$ and $E \left[ X_1 \mid X_2, Z \right]$ for both under- and over-fitting.

# ## Latent conditional independence tests
#

# ### Show the posited causal graph



# ### Main idea
#
# #### Prior distribution based test

# +
# Get the means and standard deviations of those variables
drive_alone_means = drive_alone_df.mean()
drive_alone_means.name = 'mean'

drive_alone_stds = drive_alone_df.std()
drive_alone_stds.name = 'std'

# Look at the computed means and standard deviations
print(pd.DataFrame([drive_alone_means, drive_alone_stds]).T)

# +
# Note the number of dimensions
num_dimensions = len(UTILITY_COLUMNS)

# Specify the prior distributions for the factor
# model of the standardized drive alone dataframe
w_dist_prior = scipy.stats.norm(loc=0, scale=1)
z_dist_prior = scipy.stats.norm(loc=0, scale=1)

sigma_prior = 0.1
epsilon_dist_prior =\
    scipy.stats.norm(loc=0, scale=sigma_prior)

# Get samples of x from the prior distribution factor model
x_samples_prior, z_samples_prior =\
    sample_from_factor_model(
        loadings_dist=z_dist_prior,
        coef_dist=w_dist_prior,
        noise_dist=epsilon_dist_prior,
        standard_deviations=drive_alone_stds.values,
        means=drive_alone_means.values,
        num_obs=num_drive_alone_obs,
        num_samples=NUM_PERMUTATIONS,
        num_factors=1,
        seed=721
        )

# Look at the dimensions of the prior predictive samples
print(x_samples_prior.shape)

# +
# Collect the columns being used in the test and info about them.
columns_for_test = [NUM_AUTOS_COLUMN, LICENSE_COLUMN]
col_idxs_for_test =\
    [UTILITY_COLUMNS.index(col) for col in columns_for_test]

# Get the observed values to be used for testing
obs_sample = drive_alone_df.loc[:, columns_for_test].values

# Get the prior predictive values for testing
prior_samples_triplet =\
    np.concatenate((x_samples_prior[:, col_idxs_for_test, :],
                    z_samples_prior),
                   axis=1)

# Use the predictive, conditional independence test
pval, sampled_pvals, obs_pvals =\
    li.perform_visual_predictive_cit_test(
        prior_samples_triplet,
        obs_sample)
# -

# #### Posterior distribution based test

# +
# Load the parameters of the variational approximation to 
# the posterior distribution over W and Z
w_post_params = pd.read_csv(PATH_TO_W_PARAMS, index_col=0)
z_post_params = pd.read_csv(PATH_TO_Z_PARAMS, index_col=0)

# Create the posterior distribution of coefficients
# Note we need the arguments to have shape that can,
# be broadcast to (num_factors, num_predictors, num_samples)
w_dist_post =\
    scipy.stats.norm(
        loc=w_post_params['w_mean_inferred'].values[None, : , None],
        scale=w_post_params['w_std_inferred'].values[None :, None])

# Create the posterior distribution of loadings
# Note we need the arguments to have shape that can,
# be broadcast to (num_obs, num_factors, num_samples)
z_dist_post =\
    scipy.stats.norm(
        loc=z_post_params['z_mean_inferred'].values[:, None, None],
        scale=z_post_params['z_std_inferred'].values[:, None, None])

# Get posterior samples of X_standardized
x_samples_post, z_samples_post =\
    sample_from_factor_model(
        loadings_dist=z_dist_post,
        coef_dist=w_dist_post,
        noise_dist=epsilon_dist_prior,
        standard_deviations=drive_alone_stds.values,
        means=drive_alone_means.values,
        num_obs=num_drive_alone_obs,
        num_samples=NUM_PERMUTATIONS,
        num_factors=1,
        post=False,
        seed=852
        )

# Look at the dimensions of the prior predictive samples
print(x_samples_post.shape)

# +
# Get the posterior predictive values for the test
posterior_samples_triplet =\
    np.concatenate((x_samples_post[:, col_idxs_for_test, :],
                    z_samples_post),
                   axis=1)

# Test out the predictive conditional independence test
post_pval, post_sampled_pvals, post_obs_pvals =\
    li.perform_visual_predictive_cit_test(
        posterior_samples_triplet,
        obs_sample)
# -

# ### Caveats and pitfalls



# ## Summary
