---
jupyter:
  jupytext:
    formats: ipynb,md,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Purpose

The purpose of this notebook is to demonstrate prior and predictive checks of one's causal graphical model.

The prior checks are to be used as part of one's falsification efforts before estimating the posterior distribution of one's unknown model parameters. If one's causal model contains latent variables, then such prior checks are expected to be extremely valuable. They are expected to indicate when one's model is likely to poorly fit one's data. This information can be used to avoid a potentially lengthy model estimation process. These checks will likely be implemented with very liberal thresholds for deciding that a model is not even worth beign estimated.

The posterior predictive checks are to really ensure that the observed data is well fit by the assumptions of one's causal model.

# Logical steps
0. Determine the test statistic to be computed.
1. Require as inputs:
   1. predictive samples of all model variables (latent and observed),
   2. function to compute the desired test statistic given a sample from the causal graph,
   3. the observed data.
   4. function to plot the distribution of the simulated test statistic and the value/distribution of the observed test statistic.
2. For each predictive sample,
   1. Compute the value of the simulated and observed test statistic (assuming the observed test statistic also depends on the simulated values. If not, simply store the value of the observed test statistic and do not recompute it.)
   2. Store the simulated and observed test statistics.
3. Visualize the distribution of the simulated and observed test statistics.
4. Produce a scalar summary of the distribution of simulated test statistics if desired.


## Needed refactoring
1. <strike>Pass `num_permutations` through `compute_predictive_independence_test_value` and `perform_visual_predictive_cit_test`.</strike>
2. <strike>Replace `_make_regressor` and `computed_vs_obs_r2` with their analogous functions in `testing.observable_independence`.</strike>
3. <strike>Add type hints and docstrings to `compute_pvalue`. Make sure `obs_r2` can be a scalar or 1D ndarray. `Make sure permuted_r2` is a 1D ndarray.</strike>
4. <strike>Add type hints and docstrings to all functions.</strike>
5. <strike>Add an `output_path` argument to `visualize_predictive_cit_results` and `perform_visual_predictive_cit_test` so that these functions can save the created plot to a given file location.</strike>
6. <strike>Move `predictive_cit` functions to standalone file.</strike>
7. <strike>Move causal graph to standalone file.</strike>
8. <strike>Create functions for plotting simulated CDFs in a regression context.</strike>
9. <strike>Move CDF plotting functions to standalone file.</strike>
10. Move prior and posterior sampling logic to a standalone file.
11. Write more description of what is being done in the various cells and why.


## Declare notebook parameters

```python
# Declare hyperparameters for testing
NUM_PRIOR_SAMPLES = 100

# Declare the columns to be used for testing
x1_col = 'num_cars'
x2_col = 'num_licensed_drivers'
mode_id_col = 'mode_id'

# Declare paths to data
DATA_PATH =\
    '../../data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv'
# Note that these files are based on using the `confounder`
# function from `Causal_Graph_Tim_Data.ipynb`, where the
# confounder function replicates the PPCA model of Wang
# and Blei (2018)
PATH_TO_W_PARAMS = '../../data/processed/W_inferred_PPCA.csv'
PATH_TO_Z_PARAMS = '../../data/processed/Z_inferred_PPCA.csv'
```

## Execute needed imports

```python
# Built-in modules
import sys

# Third-party modules
import numpy as np
import pandas as pd
from scipy.stats import norm

import seaborn as sbn
import matplotlib.pyplot as plt
%matplotlib inline

# Local modules
sys.path.insert(0, '../../')
import src.viz.sim_cdf as sim_cdf
import src.testing.observable_independence as oi
import src.testing.latent_independence as li
from src.graphs.drive_alone_utility import DRIVE_ALONE_UTILITY
```

## Extract data for the factor model checks

```python
# Load the raw data
df = pd.read_csv(DATA_PATH)
```

```python
# Draw the causal model motivating this test
causal_graph = DRIVE_ALONE_UTILITY.draw()
causal_graph.graph_attr.update(size="10,6")
causal_graph
```

```python
# Create a list of the variables in the drive alone utility
drive_alone_variables =\
    ['total_travel_distance',
     'total_travel_cost',
     'total_travel_time',
     'num_cars',
     'num_licensed_drivers'
    ]

# Create a sub-dataframe with those variables
drive_alone_df =\
    df.loc[df['mode_id'] == 1, drive_alone_variables]

# Get the means and standard deviations of those variables
drive_alone_means = drive_alone_df.mean()
drive_alone_means.name = 'mean'

drive_alone_stds = drive_alone_df.std()
drive_alone_stds.name = 'std'

# Look at the computed means and standard deviations
print(pd.DataFrame([drive_alone_means, drive_alone_stds]).T)
```

## Specify the factor model that is to be checked

In Wang and Blei's deconfounder technique, we fit a factor model to the variables in one's outcome model.

The factor model being considered here is:

$
\begin{aligned}
X_{\textrm{standardized}} &= Z * W + \epsilon\\
\textrm{where } \epsilon &= \left[ \epsilon_1, \epsilon_2, ..., \epsilon_D \right]\\
\epsilon_d &\in \mathbb{R}^{\textrm{N x 1}}\\
\epsilon_d &\sim \mathcal{N} \left(0, \sigma \right) \forall d \in \left\lbrace 1, 2, ... D \right\rbrace\\
Z &\in \mathbb{R}^{\textrm{N x 1}}\\
Z &\sim \mathcal{N} \left(0, 1 \right)\\
W &\in \mathbb{R}^{1 x D}\\
W &\sim \mathcal{N} \left(0, 1 \right)\\
N &= \textrm{Number of rows in X_standardized}\\
D &= \textrm{Number of columns in X_standardized}
\end{aligned}
$

```python
# Note the number of dimensions
num_dimensions = len(drive_alone_variables)

# Specify the prior distributions for the factor
# model of the standardized drive alone dataframe
w_dist_prior = norm(loc=0, scale=1)
z_dist_prior = norm(loc=0, scale=1)

sigma_prior = 0.1
epsilon_dist_prior = norm(loc=0, scale=sigma_prior)
```

## Generate prior predictive samples

```python
# Get the number of observations for this utility
num_drive_alone_obs = drive_alone_df.shape[0]

# Set a seed for reproducibility
np.random.seed(721)

# Get prior samples of X_standardized
w_samples_prior =\
    w_dist_prior.rvs((1, num_dimensions, NUM_PRIOR_SAMPLES))
z_samples_prior =\
    z_dist_prior.rvs((num_drive_alone_obs, 1, NUM_PRIOR_SAMPLES))

epsilon_samples_prior =\
    epsilon_dist_prior.rvs(size=(num_drive_alone_obs,
                                 num_dimensions,
                                 NUM_PRIOR_SAMPLES))

x_standardized_samples_prior =\
    (np.einsum('mnr,ndr->mdr', z_samples_prior, w_samples_prior) +
     epsilon_samples_prior)

# Get samples of X on the original scale of each variable
x_samples_prior =\
    (x_standardized_samples_prior *
     drive_alone_stds[None, :, None] +
     drive_alone_means[None, :, None])
```

```python
# Look at the dimensions of the prior predictive samples
print(x_samples_prior.shape)
```

## Visualize the prior predictive distribution

```python
# Choose a column of data to compare
current_col = 0

prior_sim_cdf =\
    li.plot_simulated_vs_observed_cdf(
        drive_alone_df.iloc[:, current_col].values,
        x_samples_prior[:, current_col, :],
        x_label=drive_alone_variables[current_col]
        )
```

Based on the plot above, it's clear that the currently chosen prior is quite poor.

In other words, there are highly visible levels of prior-data conflict.

This lets us know that the prior predictive check of the deconfounder assumptions is likely to fail since the prior in general is a poor one, even without considering specific checks like conditional independence tests.


### Perform the prior predictive conditional independence test

```python
# Get the values to be used for testing
obs_x1 =\
    (drive_alone_df.iloc[:, drive_alone_variables.index(x1_col)]
                   .values)
obs_x2 =\
    (drive_alone_df.iloc[:, drive_alone_variables.index(x2_col)]
                   .values)
obs_sample =\
    np.concatenate((obs_x1[:, None], obs_x2[:, None]), axis=1)
```

```python
cols_for_test =\
    [drive_alone_variables.index(col) for col in [x1_col, x2_col]]
# Get the prior predictive values for the test
prior_samples_triplet =\
    np.concatenate((x_samples_prior[:, cols_for_test, :],
                    z_samples_prior),
                   axis=1)

# Test out the predictive conditional independence test
pval, sampled_pvals, obs_pvals =\
    li.perform_visual_predictive_cit_test(
        prior_samples_triplet,
        obs_sample)
```

```python
sbn.kdeplot(sampled_pvals)
```

```python
print(obs_pvals)
```

As indicated by the observed p-values, the observed data is strongly refuted (in absolute terms) by a conditional independence test. This is shown by the p-values of zero above.

As indicated by the relative comparison of the observed p-values to the simulated p-values, the p-values generated by the observed data are very different from the p-values generated by the prior (which is known to satisfy the desired conditional independencies).

However, both of these points are somewhat moot since the prior is in general terrible.


## Posterior Predictive Conditional Independence Tests

```python
# Load the parameters of the variational approximation to 
# the posterior distribution over W and Z
w_post_params = pd.read_csv(PATH_TO_W_PARAMS, index_col=0)
z_post_params = pd.read_csv(PATH_TO_Z_PARAMS, index_col=0)
```

```python
w_post_params['w_var_inferred'] = w_post_params['w_std_inferred']**2
w_post_params
```

```python
# Create the posterior distribution of
w_dist_post =\
    norm(loc=w_post_params['w_mean_inferred'].values,
         scale=w_post_params['w_std_inferred'].values)

z_dist_post =\
    norm(loc=z_post_params['z_mean_inferred'].values,
         scale=z_post_params['z_std_inferred'].values)
```

```python
# Set a seed for reproducibility
np.random.seed(852)

# Get posterior samples of X_standardized
w_samples_post =\
    w_dist_prior.rvs((num_dimensions, NUM_PRIOR_SAMPLES))
z_samples_post =\
    z_dist_prior.rvs((num_drive_alone_obs, NUM_PRIOR_SAMPLES))

# Convert the posterior samples to have the desired shapes
w_samples_post = w_samples_post[None, :, :]
z_samples_post = z_samples_post[:, None, :]

# Sample epsilon to form the posterior samples of X_standardized
epsilon_samples_post =\
    epsilon_dist_prior.rvs(size=(num_drive_alone_obs,
                                 num_dimensions,
                                 NUM_PRIOR_SAMPLES))

x_standardized_samples_post =\
    (np.einsum('mnr,ndr->mdr', z_samples_post, w_samples_post) +
     epsilon_samples_post)

# Get samples of X on the original scale of each variable
x_samples_post =\
    (x_standardized_samples_post *
     drive_alone_stds[None, :, None] +
     drive_alone_means[None, :, None])
```

```python
# Look at the dimensions of the prior predictive samples
print(x_samples_post.shape)
```

### Visualize the posterior predictive distribution

```python
# Choose a column of data to compare
current_col = 0

posterior_sim_cdf =\
    li.plot_simulated_vs_observed_cdf(
        drive_alone_df.iloc[:, current_col].values,
        x_samples_post[:, current_col, :],
        x_label=drive_alone_variables[current_col]
        )
```

```python
# Compare a single sample from the prior and posterior distributions
# to the observed data
total_travel_dist_samples =\
    pd.DataFrame({'total_travel_distance_prior': x_samples_prior[:, 0, 0],
                  'total_travel_distance_post': x_samples_post[:, 0, 0],
                  'total_travel_distance_obs':
                      drive_alone_df['total_travel_distance'].values})

total_travel_dist_samples.describe()
```

The plot above summarizes the posterior distribution of the total travel distance.
Similar to the prior distribution of the same variable, the posterior poorly fits the data.
As before, we can immediately expect the posterior predictive version of the conditional independence to fail since the observed data is generally unlike the sampled data.
This is dissimilarity is, a-priori, expected to remain in the conditional independence test.

```python
# Get the prior predictive values for the test
posterior_samples_triplet =\
    np.concatenate((x_samples_post[:, cols_for_test, :],
                    z_samples_post),
                   axis=1)

# Test out the predictive conditional independence test
post_pval, post_sampled_pvals, post_obs_pvals =\
    li.perform_visual_predictive_cit_test(
        posterior_samples_triplet,
        obs_sample)
```

```python
sbn.kdeplot(post_sampled_pvals)
```

```python
print(post_obs_pvals)
```

# Test the predictive conditional independence tests
Make sure that the predictive condidtional independence tests are passed when using data that we know satisfies the independence assumptions being tested

```python
chosen_sim_idx = 50

# Test the predictive C.I.T with a prior sample
prior_sim_sample = x_samples_prior[:, cols_for_test, chosen_sim_idx]
prior_pval_sim, prior_sampled_pvals_sim, prior_obs_pvals_sim =\
    li.perform_visual_predictive_cit_test(
        prior_samples_triplet,
        prior_sim_sample)
```

```python
prior_sampled_pvals_sim
```

```python
prior_obs_pvals_sim
```

```python
# Test the predictive C.I.T with a posterior sample
post_sim_sample = x_samples_post[:, cols_for_test, chosen_sim_idx]
post_pval_sim, post_sampled_pvals_sim, post_obs_pvals_sim =\
    li.perform_visual_predictive_cit_test(
        posterior_samples_triplet,
        post_sim_sample)
```

```python
post_sampled_pvals_sim
```

```python
post_obs_pvals_sim
```

# Conclusions
From the results above, a few things are apparent.

1. The prior distribution for this particular implementation of the deconfounder is a very poor description of reality. A-priori, our prior beliefs are in severe conflict with our data and are likely in need or revising to be more plausible. For instance, we should never be simulating negative values for `total_travel_distance`.
2. The posterior distribution for this particular implementation of the deconfounder is still a poor description of our data.
3. In order for the predictive conditional independence tests to pass, the inferred latent confounder values must be extremely close to the true latent confounder values. This provides a secondary piece of evidence supporting the finding from Hassan's deconfounder investigation.
   1. The only instance of the test of the deconfounder that succeeded in generating a non-zero p-value (see cell above) is the instance where the value being used as the "observation" was paired with its own latent confounders.
   2. In order for many instances of the test of the deconfounder to succeed in generating non-zero p-values, we'd need most of the simulated latent confounder values to cluster around their true latent confounder values.
