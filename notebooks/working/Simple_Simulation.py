# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel, StructuralCausalModel
import pylogit
from collections import OrderedDict
import pylogit as cm
from functools import reduce
import statsmodels.api as sm

from IPython import display

import os
os.listdir('.')
# -

data = pd.read_csv('spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv')
data.columns

# +
### Just look at the drive alone mode
drive_alone_df = data[data['mode_id']==1]

### Only keep columns of interest 
drive_alone_df = drive_alone_df[[ 'total_travel_time', 'total_travel_cost',
                                 'total_travel_distance', 'household_size',
                               'num_cars', 'cross_bay']]

drive_alone_df.describe()

# -

# ## Assumed causal graph

# +
drive_alone_graph = CausalGraphicalModel(
    nodes=['total_travel_time', 'total_travel_cost', 'total_travel_distance', #'household_income',
          'household_size',  'num_cars', 'cross_bay', 'utility_driving'],
    edges=[
        ("total_travel_time", "utility_driving"), 
        ("total_travel_cost", "utility_driving"), 
        ("total_travel_distance", "utility_driving"), 
        ("household_size", "utility_driving"), 
#         ("household_income", "utility_driving"), 
        ("num_cars", "utility_driving"), 
        ("cross_bay", "utility_driving"), 
        
        
        ("total_travel_distance", "total_travel_time"), 
        ("total_travel_distance", "total_travel_cost"), 
    ]
)

# draw return a graphviz `dot` object, which jupyter can render
drive_alone_graph.draw()


# -

# ## Distributional regression
# Assume univariate linear approximation for the relationship between travel distance and travel time/cost. Turns out it's not a horrible assumption.

def fit_regression(X, y, data, plotting = True):
    data_x = sm.add_constant(data[X])
    data_y = data[y]

    model = sm.OLS(data_y, data_x)
    
    results = model.fit()
    
    if plotting:
        fig = plt.figure(figsize=(12,8))
        fig = sm.graphics.plot_regress_exog(results, X[0], fig=fig)
    
    return results


# +
## Regress travel cost on travel distance

cost_on_distance_reg = fit_regression(X=['total_travel_distance'], 
                                      y = ['total_travel_cost'], data=drive_alone_df, plotting = True)

cost_on_distance_reg.summary()

# +
## Regress travel time on travel distance

time_on_distance_reg = fit_regression(X=['total_travel_distance'], 
                                      y = ['total_travel_time'], data=drive_alone_df, plotting = True)

time_on_distance_reg.summary()
# -

# ### Simulate travel time and cost based on the distributional regression

# +
## residuals spread -- assuming homoscedasticity
time_on_dist_std = time_on_distance_reg.resid.std()
cost_on_dist_std = cost_on_distance_reg.resid.std()

sample_size = len(drive_alone_df)

# +
total_travel_time_sim = ( time_on_distance_reg.params[0] + 
                         time_on_distance_reg.params[1]*drive_alone_df['total_travel_distance']
                        + np.random.normal(loc=0, scale=time_on_dist_std, size = sample_size) )

total_travel_cost_sim = ( cost_on_distance_reg.params[0] + 
                         cost_on_distance_reg.params[1]*drive_alone_df['total_travel_distance']
                        + np.random.normal(loc=0, scale=cost_on_dist_std, size = sample_size) )
# -

# ### Create a simulation dataframe

simulation_df = drive_alone_df.copy()
simulation_df['total_travel_time'] = total_travel_time_sim
simulation_df['total_travel_cost'] = total_travel_cost_sim

# ### Simulate the utility equation based on hypothetical parameters

simulation_df.columns

## Assumed coefficients for above columns
coeffs = np.array([-.5, -1.5, 3, 5, 2, -7 ])
asc_drive = 6.

# +
utilities = asc_drive + np.dot( simulation_df, coeffs) + np.random.normal(loc=0, scale = 1, size = sample_size)

simulation_df['utilities'] = utilities
# -

# ## Estimation
#
# Note that here, I'm treating the utilities as an observed quantity that I'm trying to estimate. This will get more complicated as we include different modes and actually try to maximize the correct likelihood function. 
#
# The thing I need to point out here is that irrespective of our causal graph, we will always recover the paramters in coeffs (defined above) when we run the regression. The question becomes, are the coefficients on each of the variables (the true coefficients) causal? That depends on the causal graph:
#
# - In the case of independent variables, the true causal effect of distance is the 0.1, the same value in the coeffs array. Thus, running a regression on all the variables in this case would return the true causal estimate. 
#
# - In the case where travel cost and travel time are descendents of distance, the true causal effect of distance becomes: 0.1 - 0.5*(1.28) - 1.5*(0.22) = -0.87. We will only recover this value if we omit travel distance and travel time from the utility equation. Alternatively, we can keep them in the equation, but then don't assign the coefficient for distance any causal interpretation, and post-process the results (and make parametric assumptions!) to calculate the true causal effect. 

## Regress utilities on all the covariates. Notice how we recover the simulation parameters.
## The coefficient on travel distance however, is not causal
utilities_regress = fit_regression(X=list(simulation_df.columns[:-1]), 
                                      y = ['utilities'], data=simulation_df, plotting = False)
utilities_regress.summary()

# +
## Now regress utilities on all the covariates except for travel time and cost. 
## The coefficient on travel distance now is causal. However, this is not a good model of the output. 
X = [ 'total_travel_distance',
       'household_size', 'num_cars', 'cross_bay'
    ]

utilities_regress_causal = fit_regression(X=X, 
                                      y = ['utilities'], data=simulation_df, plotting = False)
utilities_regress_causal.summary()
# -

# ## What if we don't observe travel distance?

# ### Fit PCA

# +
import numpy as np
from numpy.random import randn, rand
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from nnls import nnlsm_blockpivot as nnlstsq
import itertools
from scipy.spatial.distance import cdist

def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B
    Note: uses a broadcasted solve for speed.
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    if A.ndim == 1:
        A = A[:,None]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        # transpose to get r x n
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T
    except:
        r = T.shape[1]
        T[:,np.arange(r),np.arange(r)] += 1e-6
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T



def cv_pca(data, rank, M=None, p_holdout=0.3, nonneg=False, iterations = 1000):
    """Fit PCA while holding out a fraction of the dataset.
    """

#     # choose solver for alternating minimization
#     if nonneg:
#         solver = censored_nnlstsq
#     else:
    solver = censored_lstsq

    # create masking matrix
    if M is None:
        M = np.random.rand(*data.shape) > p_holdout

    # initialize U randomly
    if nonneg:
        U = np.random.rand(data.shape[0], rank)
    else:
        U = np.random.randn(data.shape[0], rank)

    # fit pca/nmf
    for itr in range(iterations):
        Vt = solver(U, data, M)
        U = solver(Vt.T, data.T, M.T).T

    # return result and test/train error
    resid = np.dot(U, Vt) - data
    train_err = np.mean(resid[M]**2)
    test_err = np.mean(resid[~M]**2)
    return U, Vt, train_err, test_err, M, resid


# +
X_columns = [
    'total_travel_time',
       'total_travel_cost', 
            ]


X = np.array((simulation_df[X_columns] - simulation_df[X_columns].mean())/simulation_df[X_columns].std())


# X_raw = np.array([s2,s3]).reshape((1000,2))
# X =( X_raw - X_raw.mean(axis=0) )/X_raw.std(axis=0)


U, Vt, train_err, test_err, M, resid = cv_pca(data=X, rank=2)
train_err, test_err
# -

# ### Check PCA

# +
fig, ax = plt.subplots()
display.display(pd.Series(resid[:,0]).hist(bins=50))

# fig, ax = plt.subplots()
display.display(pd.Series(resid[:,1]).hist(bins=50))
# -

# ## Include confounder in regression

simulation_df['confounder_PCA'] = U[:,1]

# +
X_conf = ['total_travel_time', 'total_travel_cost', 
       'household_size', 'num_cars', 'cross_bay', 
       'confounder_PCA']


utilities_regress = fit_regression(X=X_conf, 
                                      y = ['utilities'], data=simulation_df, plotting = False)
utilities_regress.summary()

# +
X_true = ['total_travel_time', 'total_travel_cost', 
       'household_size', 'num_cars', 'cross_bay', 
       'total_travel_distance']


utilities_regress = fit_regression(X=X_true, 
                                      y = ['utilities'], data=simulation_df, plotting = False)
utilities_regress.summary()

# +
X_ommitted = ['total_travel_time', 'total_travel_cost', 
       'household_size', 'num_cars', 'cross_bay', 
       ]


utilities_regress = fit_regression(X=X_ommitted, 
                                      y = ['utilities'], data=simulation_df, plotting = False)
utilities_regress.summary()
# -

# # Scratch

# +
scratch = CausalGraphicalModel(
    nodes=['a', 'b', 'c', 'y'],
    edges=[
        ("b", "a"), 
        ("b", "c"), 
        ("b", "y"), 
        ("a", "y"), 
        ("c", "y"), 
 
 
    ]
)

scratch.draw()

# +
size = 2000

b = np.random.normal(loc=10, scale = 2, size = size)

a = np.random.normal(loc=0, scale = 1, size = size) + 2  + 3*b 

c =  np.random.normal(loc=0, scale = 1, size = size) - 3 - 7*b 

y = 6 - 7*b + 3*a -2*c + np.random.normal(loc=0, scale = 1, size = size)

# +
regress_df = pd.DataFrame()

regress_df['a'] = a
regress_df['b'] = b
regress_df['c'] = c
regress_df['y'] = y

mod_scratch = sm.OLS(regress_df['y'], sm.add_constant(regress_df[['b']]))
res = mod_scratch.fit()

res.summary()
# -




