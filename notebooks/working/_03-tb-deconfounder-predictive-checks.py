# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py
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

# # Purpose
#
# The purpose of this notebook is to demonstrate prior and predictive checks of one's causal graphical model.
#
# The prior checks are to be used as part of one's falsification efforts before estimating the posterior distribution of one's unknown model parameters. If one's causal model contains latent variables, then such prior checks are expected to be extremely valuable. They are expected to indicate when one's model is likely to poorly fit one's data. This information can be used to avoid a potentially lengthy model estimation process. These checks will likely be implemented with very liberal thresholds for deciding that a model is not even worth beign estimated.
#
# The posterior predictive checks are to really ensure that the observed data is well fit by the assumptions of one's causal model.
#
# # Logical steps
# 0. Determine the test statistic to be computed.
# 1. Require as inputs:
#    1. predictive samples of all model variables (latent and observed),
#    2. function to compute the desired test statistic given a sample from the causal graph,
#    3. the observed data.
#    4. function to plot the distribution of the simulated test statistic and the value/distribution of the observed test statistic.
# 2. For each predictive sample,
#    1. Compute the value of the simulated and observed test statistic (assuming the observed test statistic also depends on the simulated values. If not, simply store the value of the observed test statistic and do not recompute it.)
#    2. Store the simulated and observed test statistics.
# 3. Visualize the distribution of the simulated and observed test statistics.
# 4. Produce a scalar summary of the distribution of simulated test statistics if desired.


