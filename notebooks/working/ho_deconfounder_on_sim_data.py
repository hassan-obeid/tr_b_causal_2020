# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Purpose
#

# The purpose of this notebook is to investigate the effectiveness of the deconfounder algorithm (Blei et. al, 2018) in adjusting for unobserved confounding. We use a simulated mode choice data where travel distance linearly confounds both travel time and travel cost. We then mask the travel distance data and treat it as an unobserved variable. 
#
# We estimate three models:
# - Model 1: A multinomial logit with the correct original specification, EXCEPT we ommit the travel distance variable in the specification without trying to adjust for it. 
# - Model 2: We use the deconfounder algorithm to try to recover the confounder (travel distance). In this method, we use all the variables in each mode's utility to recover that mode's confounder.
# - Model 3: We use the deconfounder algorithm to try to recover the confounder (travel distance), but this time, we only use travel time and cost in the factor model, instead of all the variables in the utility specification of each mode. 
#
# We compare the estimates of the coefficients on travel time and cost to the true estimates used in the simulation. The main findings of this exercise are the following:
# - Using the true variables believed to be confounded (i.e. method 3 where only travel time and cost are used to recover the confounder) leads to a better recovery of the true confounder. This suggests that it may be better to run the deconfounder algorithm based on a hypothesized causal graph, rather than just running it on all the observed covariates. 
# - Similar to what we found in the investigating_decounfounder notebook, the effectiveness of the deconfounder algorithm is very sensitive to small deviations in the recovered confounder. Although method 3 returns a relatively good fit of the true confounder, the adjusted coefficients on travel time and cost do not exhibit any reduction in the bias resulting from ommitting the true confounder, and the coefficients on the recovered confounder are highly insignificant. This raises questions about the usefulness of the deconfounder algorithm in practice.

# # Import needed libraries

# +
# Built-in modules
import os
from collections import OrderedDict
from functools import reduce

# Third party modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel, StructuralCausalModel
import pylogit as cm

# Local modules
from factor_models import *

# -

# ## Useful function

def specifications(mnl_specification, num_modes):
    newDict= dict()

    for i in range(1,num_modes+1):
        variables = []
        # Iterate over all the items in dictionary and filter items which has even keys
        for (key, value) in mnl_specification.items():
           # Check if key is even then add pair to new dictionary
    



            if any(isinstance(sub, list) for sub in value):
                for sub in value:
                    if isinstance(sub, list):
                        if (i in sub): 
                            variables.append(key)
                            
                    else:
                        if i == sub:
                            variables.append(key)

            else:
                if i in value:
        #             print(variables)

                    variables.append(key)

        newDict[i] = variables

    return newDict

# ## Analysis

PATH = '../../data/raw/'
file_name = 'simulated_long_format_bike_data.csv'

data = pd.read_csv(PATH+file_name)
data = data.drop('Unnamed: 0', axis = 1)
data.columns

# +
X_columns = ['total_travel_time',
       'total_travel_cost', 'total_travel_distance', 
             'cross_bay', 'household_size', 'num_kids', 
              'cars_per_licensed_drivers', 
             'gender'
             
            ]

y_column = data['mode_id']

# +
## Specify the true causal graph used in the simulation of the choices

causal_graph = CausalGraphicalModel(
    nodes=["Travel_Time", "Travel_Distance", "Travel_Cost", "Cross_Bay_Bridge", "HH_Size", "num_of_kids_household",
          "Autos_per_licensed_drivers", "Gender", "Mode_Choice"],
    edges=[
        ("Travel_Time", "Mode_Choice"), 
        ("Travel_Distance", "Mode_Choice"), 
        ("Travel_Cost", "Mode_Choice"), 
        ("Cross_Bay_Bridge", "Mode_Choice"), 
        ("HH_Size", "Mode_Choice"), 
        ("num_of_kids_household", "Mode_Choice"), 
        ("Autos_per_licensed_drivers", "Mode_Choice"), 
        ("Gender", "Mode_Choice"), 
        
        
        ("Travel_Distance", "Travel_Time"), 
        ("Travel_Distance", "Travel_Cost"), 
        ("Travel_Distance", "Cross_Bay_Bridge"), 
        ("HH_Size", "Travel_Distance"), 

        
        
#         ("Travel_Time", "Mode_Choice"), 
#         ("Travel_Time", "Mode_Choice"), 

        
        
    ]
)

# draw return a graphviz `dot` object, which jupyter can render
causal_graph.draw()
# -

# # MNL specification

# +
## Below is the specification of the true model used in the simulation of the choices

mnl_specification = OrderedDict()
mnl_names = OrderedDict()

mnl_specification["intercept"] = [2, 3, 4, 5, 6, 7, 8]
mnl_names["intercept"] = ['ASC Shared Ride: 2',
                          'ASC Shared Ride: 3+',
                          'ASC Walk-Transit-Walk',
                          'ASC Drive-Transit-Walk',
                          'ASC Walk-Transit-Drive',
                          'ASC Walk',
                          'ASC Bike']

mnl_specification["total_travel_time"] = [1, 2, 3, [4, 5, 6]]
mnl_names["total_travel_time"] = ['Travel Time, units:min (Drive Alone)',
                                  'Travel Time, units:min (SharedRide-2)',
                                  'Travel Time, units:min (SharedRide-3+)',
                                  'Travel Time, units:min (All Transit Modes)']

mnl_specification["total_travel_cost"] = [1, 2, 3, [4, 5, 6]]
mnl_names["total_travel_cost"] = ['Travel Cost, units:$ (Drive Alone)',
                                  'Travel Cost, units:$ (SharedRide-2)',
                                  'Travel Cost, units:$ (SharedRide-3+)',
                                  'Travel Cost, units:$ (All Transit Modes)']

# mnl_specification["cost_per_distance"] = [1, 2, 3]
# mnl_names["cost_per_distance"] = ["Travel Cost per Distance, units:$/mi (Drive Alone)",
#                                   "Travel Cost per Distance, units:$/mi (SharedRide-2)",
#                                   "Travel Cost per Distance, units:$/mi (SharedRide-3+)"]

mnl_specification["cars_per_licensed_drivers"] = [[1, 2, 3]]
mnl_names["cars_per_licensed_drivers"] = ["Autos per licensed drivers (All Auto Modes)"]

mnl_specification["total_travel_distance"] = [1, 2, 3, 7, 8]
mnl_names["total_travel_distance"] = ['Travel Distance, units:mi (Drive Alone)',
                                      'Travel Distance, units:mi (SharedRide-2)',
                                      'Travel Distance, units:mi (SharedRide-3+)',
                                      'Travel Distance, units:mi (Walk)',
                                      'Travel Distance, units:mi (Bike)']

# mnl_specification["cross_bay"] = [[2, 3], [4, 5, 6]]
# mnl_names["cross_bay"] = ["Cross-Bay Tour (Shared Ride 2 & 3+)",
#                           "Cross-Bay Tour (All Transit Modes)"]
mnl_specification["cross_bay"] = [[2, 3]]
mnl_names["cross_bay"] = ["Cross-Bay Tour (Shared Ride 2 & 3+)"]

mnl_specification["household_size"] = [[2, 3]]
mnl_names["household_size"] = ['Household Size (Shared Ride 2 & 3+)']

mnl_specification["num_kids"] = [[2, 3]]
mnl_names["num_kids"] = ["Number of Kids in Household (Shared Ride 2 & 3+)"]
# -

# # Recovering the confounder: The deconfounder algorithm

# ## First approach: using all the variables in the factor model
#
# Using this approach, we use all the variables in each mode's utility specification to recover the confounder, in line with Blei et al.'s approach in their paper.

# +
## Return the variables of each utility specification

spec_dic = specifications(mnl_specification=mnl_specification, num_modes=8)
spec_dic
# -

confounder_vectors = []
holdout_dfs = []
masks_df = []
rows_df =[]
latent_dim = 1
for i in data['mode_id'].unique():

    data_mode_i = data[data['mode_id']==i]
    # standardize the data for PPCA
    print("Analysis for mode: ", i)
    print("-------------------------------------------------------------------------------------------")
    X_columns = spec_dic[i]
    
    if 'intercept' in X_columns:
        X_columns.remove('intercept')
        
    if 'total_travel_distance' in X_columns:
        X_columns.remove('total_travel_distance')
        
    print(i, X_columns)
    
    X = np.array((data_mode_i[X_columns] - data_mode_i[X_columns].mean())/data_mode_i[X_columns].std())
    
    confounders, holdouts, holdoutmasks, holdoutrow= confounder_ppca(holdout_portion=0.2, X=X, latent_dim=latent_dim)

    confounder_vectors.append(confounders)
    holdout_dfs.append(holdouts)
    masks_df.append(holdoutmasks)
    rows_df.append(holdoutrow)

# ## Second approach: using only the confounded variables in the factor model
# Using this approach, we only run the deconfounder's factor model on travel_time and travel_cost, which are the only variables confounded with travel_distance in our simulation

# +
confounded_variables = ['total_travel_time', 'total_travel_cost']
X_DA = (data[data['mode_id']==1][confounded_variables] - 
        data[data['mode_id']==1][xs].mean())/data[data['mode_id']==1][xs].std()

X_SR2 = ((data[data['mode_id']==2][confounded_variables] - 
         data[data['mode_id']==2][confounded_variables].mean())
         /data[data['mode_id']==2][confounded_variables].std())
X_SR3 = ((data[data['mode_id']==3][confounded_variables] - 
         data[data['mode_id']==3][confounded_variables].mean())
         /data[data['mode_id']==3][confounded_variables].std())

(confounders_DA, holdouts_DA, 
 holdoutmasks_DA, holdoutrow_DA)= confounder_ppca(holdout_portion=0.2, 
                                       X=X_DA, latent_dim=latent_dim)

(confounders_SR2, holdouts_SR2, 
 holdoutmasks_SR2, holdoutrow_SR2)= confounder_ppca(holdout_portion=0.2, 
                                       X=X_SR2, latent_dim=latent_dim)

(confounders_SR3, holdouts_SR3, 
 holdoutmasks_SR3, holdoutrow_SR3)= confounder_ppca(holdout_portion=0.2, 
                                       X=X_SR3, latent_dim=latent_dim)




# -

# ## Adding confounders to original DF

# +
def add_confounders_to_df(data, confounder_vectors):
    for i in data['mode_id'].unique():
    
#     print(len(data.loc[data['mode_id']==i, col_name]), len(confounder_vectors[int(i-1)][2]) )
    
        col_name = 'confounder_for_mode_' + str(int(i))
        data.loc[data['mode_id']==i, col_name] = confounder_vectors[int(i-1)][2]
        data[col_name] = data[col_name].fillna(0)

    data['confounder_all'] = data[['confounder_for_mode_1','confounder_for_mode_2','confounder_for_mode_3',
                                  'confounder_for_mode_4', 'confounder_for_mode_5', 'confounder_for_mode_6',
                                  'confounder_for_mode_7', 'confounder_for_mode_8']].sum(axis=1)


# +
for i in data['mode_id'].unique():
    
#     print(len(data.loc[data['mode_id']==i, col_name]), len(confounder_vectors[int(i-1)][2]) )
    
    col_name = 'confounder_for_mode_' + str(int(i))
    data.loc[data['mode_id']==i, col_name] = confounder_vectors[int(i-1)][2]
    data[col_name] = data[col_name].fillna(0)
    
data['confounder_all'] = data[['confounder_for_mode_1','confounder_for_mode_2','confounder_for_mode_3',
                              'confounder_for_mode_4', 'confounder_for_mode_5', 'confounder_for_mode_6',
                              'confounder_for_mode_7', 'confounder_for_mode_8']].sum(axis=1)


data.loc[data['mode_id']==1, 'confounder_da'] = confounders_DA[2]
data['confounder_da'] = data['confounder_da'].fillna(0)

data.loc[data['mode_id']==2, 'confounder_sr2'] = confounders_SR2[2]
data['confounder_sr2'] = data['confounder_sr2'].fillna(0)

data.loc[data['mode_id']==3, 'confounder_sr3'] = confounders_SR3[2]
data['confounder_sr3'] = data['confounder_sr3'].fillna(0)

data['confounder_all_2'] = data[['confounder_da', 'confounder_sr2', 'confounder_sr3']].sum(axis=1)
data.head()

# -

# # Compare recovered confounder to actual confounder
#
# Notice here that although the recovered confounder correlates well with the true confounder, there is more noise in method 1 where we use all the variables in each utility to recover the confounder. Method 2 has less noise, but as we will see later in the notebook, this noise is still high enough that our estimates on travel_time and travel_cost will still be biased even after adjusting for the recovered confounder.

# +
data_da = data[data['mode_id']==1]

data_da.plot.scatter('total_travel_distance', 'confounder_all')

data_da.plot.scatter('total_travel_distance', 'confounder_all_2')
# -

# ## True Model

# +
# Estimate the basic MNL model, using the hessian and newton-conjugate gradient
mnl_model = cm.create_choice_model(data=data,
                                   alt_id_col="mode_id",
                                   obs_id_col="observation_id",
                                   choice_col="sim_choice",
                                   specification=mnl_specification,
                                   model_type="MNL",
                                   names=mnl_names)

num_vars = len(reduce(lambda x, y: x + y, mnl_names.values()))
mnl_model.fit_mle(np.zeros(num_vars),
                  method="BFGS")
mnl_model.get_statsmodels_summary()
# -

# ## Model 1

# +
# Create my specification and variable names for the basic MNL model
# NOTE: - Keys should be variables within the long format dataframe.
#         The sole exception to this is the "intercept" key.
#       - For the specification dictionary, the values should be lists
#         or lists of lists. Within a list, or within the inner-most
#         list should be the alternative ID's of the alternative whose
#         utility specification the explanatory variable is entering.

mnl_specification_noncausal = OrderedDict()
mnl_names_noncausal = OrderedDict()

mnl_specification_noncausal["intercept"] = [2, 3, 4, 5, 6, 7, 8]
mnl_names_noncausal["intercept"] = ['ASC Shared Ride: 2',
                          'ASC Shared Ride: 3+',
                          'ASC Walk-Transit-Walk',
                          'ASC Drive-Transit-Walk',
                          'ASC Walk-Transit-Drive',
                          'ASC Walk',
                          'ASC Bike']

mnl_specification_noncausal["total_travel_time"] = [1, 2, 3, [4, 5, 6]]
mnl_names_noncausal["total_travel_time"] = ['Travel Time, units:min (Drive Alone)',
                                  'Travel Time, units:min (SharedRide-2)',
                                  'Travel Time, units:min (SharedRide-3+)',
                                  'Travel Time, units:min (All Transit Modes)']

mnl_specification_noncausal["total_travel_cost"] = [1, 2, 3, [4, 5, 6]]
mnl_names_noncausal["total_travel_cost"] = ['Travel Cost, units:$ (Drive Alone)',
                                  'Travel Cost, units:$ (SharedRide-2)',
                                  'Travel Cost, units:$ (SharedRide-3+)',
                                  'Travel Cost, units:$ (All Transit Modes)']

mnl_specification_noncausal["cars_per_licensed_drivers"] = [[1, 2, 3]]
mnl_names_noncausal["cars_per_licensed_drivers"] = ["Autos per licensed drivers (All Auto Modes)"]

mnl_specification_noncausal["cross_bay"] = [[2, 3]]
mnl_names_noncausal["cross_bay"] = ["Cross-Bay Tour (Shared Ride 2 & 3+)"]

mnl_specification_noncausal["household_size"] = [[2, 3]]
mnl_names_noncausal["household_size"] = ['Household Size (Shared Ride 2 & 3+)']

mnl_specification_noncausal["num_kids"] = [[2, 3]]
mnl_names_noncausal["num_kids"] = ["Number of Kids in Household (Shared Ride 2 & 3+)"]

# +
# Estimate the basic MNL model, using the hessian and newton-conjugate gradient
mnl_model_noncausal = cm.create_choice_model(data=data,
                                   alt_id_col="mode_id",
                                   obs_id_col="observation_id",
                                   choice_col="sim_choice",
                                   specification=mnl_specification_noncausal,
                                   model_type="MNL",
                                   names=mnl_names_noncausal)

num_vars_noncausal = len(reduce(lambda x, y: x + y, mnl_names_noncausal.values()))
mnl_model_noncausal.fit_mle(np.zeros(num_vars_noncausal),
                  method="BFGS")

# Look at the estimation results
mnl_model_noncausal.get_statsmodels_summary()
# -

# ## Model 2

# +
mnl_specification_causal_1 = mnl_specification_noncausal.copy()
mnl_names_causal_1 = mnl_names_noncausal.copy()

mnl_specification_causal_1["confounder_all"] = [1, 2, 3]
mnl_names_causal_1["confounder_all"] = ["Confounder - Drive alone",
                                     "Confounder - Shared ride 2", 
                                     "Confounder - Shared ride 3"]

# +
# Estimate the basic MNL model, using the hessian and newton-conjugate gradient
mnl_model_causal_1 = cm.create_choice_model(data=data,
                                   alt_id_col="mode_id",
                                   obs_id_col="observation_id",
                                   choice_col="sim_choice",
                                   specification=mnl_specification_causal_1,
                                   model_type="MNL",
                                   names=mnl_names_causal_1)

num_vars = len(reduce(lambda x, y: x + y, mnl_names_causal_1.values()))
mnl_model_causal_1.fit_mle(np.zeros(num_vars),
                  method="BFGS")

# Look at the estimation results
mnl_model_causal_1.get_statsmodels_summary()
# -

# ## Model 3

# +
mnl_specification_causal_2 = mnl_specification_noncausal.copy()
mnl_names_causal_2 = mnl_names_noncausal.copy()

mnl_specification_causal_2["confounder_all_2"] = [1, 2, 3]
mnl_names_causal_2["confounder_all_2"] = ["Confounder - Drive alone",
                                     "Confounder - Shared ride 2", 
                                     "Confounder - Shared ride 3"]

# +
# Estimate the basic MNL model, using the hessian and newton-conjugate gradient
mnl_model_causal_2 = cm.create_choice_model(data=data,
                                   alt_id_col="mode_id",
                                   obs_id_col="observation_id",
                                   choice_col="sim_choice",
                                   specification=mnl_specification_causal_2,
                                   model_type="MNL",
                                   names=mnl_names_causal_2)

num_vars = len(reduce(lambda x, y: x + y, mnl_names_causal_2.values()))
mnl_model_causal_2.fit_mle(np.zeros(num_vars),
                  method="BFGS")

# Look at the estimation results
mnl_model_causal_2.get_statsmodels_summary()
# -

# ## Compare estimates on travel time and cost

# +
results_summary_true = mnl_model.get_statsmodels_summary()
results_summary_non_causal = mnl_model_noncausal.get_statsmodels_summary()
results_summary_method_1 = mnl_model_causal_1.get_statsmodels_summary()
results_summary_method_2 = mnl_model_causal_2.get_statsmodels_summary()

results_as_html_true = results_summary_true.tables[1].as_html()
results_as_html_true = pd.read_html(results_as_html_true, header=0, index_col=0)[0]

results_as_html_noncausal = results_summary_non_causal.tables[1].as_html()
results_as_html_noncausal = pd.read_html(results_as_html_noncausal, header=0, index_col=0)[0]

results_as_html_method_1 = results_summary_method_1.tables[1].as_html()
results_as_html_method_1 = pd.read_html(results_as_html_method_1, header=0, index_col=0)[0]

results_as_html_method_2 = results_summary_method_2.tables[1].as_html()
results_as_html_method_2 = pd.read_html(results_as_html_method_2, header=0, index_col=0)[0]

# +
locs = ['Travel Time, units:min (Drive Alone)','Travel Time, units:min (SharedRide-2)',
       'Travel Time, units:min (SharedRide-3+)', 'Travel Cost, units:$ (Drive Alone)',
       'Travel Cost, units:$ (SharedRide-2)', 'Travel Cost, units:$ (SharedRide-3+)']
cols = ['coef', 'std err']

results_comparison = results_as_html_true.loc[locs][cols].join(results_as_html_noncausal.loc[locs][cols], 
                                                               lsuffix = '_true', rsuffix = '_non_causal').join(
    results_as_html_method_1.loc[locs][cols].join(results_as_html_method_2.loc[locs][cols],
                                               lsuffix = '_method_1', rsuffix = '_method_2'))

results_comparison
# -













