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

# # Purpose
# This notebook will provide an illustration of a discrete choice model that tries to estimate causal effects, without making use of the framework outlined in Brathwaite and Walker (2018).
# The study of note is
# > Chiara, Giacomo Dalla, Lynette Cheah, Carlos Lima Azevedo, and Moshe E. Ben-Akiva. "A Policy-Sensitive Model of Parking Choice for Commercial Vehicles in Urban Areas." Transportation Science (2020).
#
# This notebook draw the causal diagram that we believes corresponds to the parking choice model in that paper.
# We then point out some tests that could have been done and should be presented to justify this paper's causal model and its implicit assumptions, even before estimating parameters of the proposed choice models.
# Finally, we will:
# - note the importance of these implicit assumptions, and
# - show this study may present a real example of our presentations simplified demsonstration of how incorrect causal graphs can lead to highly inaccurate causal effect estimates, even in selection-on-observables settings.
#

from causalgraphicalmodels import CausalGraphicalModel

# +
# Note that string representations stored
# as objects so I can later easily change the
# string representations in one place and update
# the representation everywhere
parking_duration = 'parking_duration\n(d_n)'
vehicle_type = 'vehicle_type\n(v_n)'
choice_set = 'choice_set\n(C_n)'
parking_tariffs = 'parking_tariffs'
parking_fines = 'parking_fines'
parking_cost = 'parking_cost\n(C_ni)'
vehicle_attributes = 'vehicle_attributes\n(Z_n)'
parking_congestion = 'parking_congestion\n(q_it)'
parking_capacity = 'parking_capacity\n(N_i)'
explanatory_features = 'explanatory_features\n(X_nit)'
unobserved_confounders = 'unobserved_confounders'
utility = 'utility\n(U_nit)'
availability = 'availability\n(A_nit)'
choice = 'choice\n(Y_nit)'

nodes =\
    [parking_duration,
     vehicle_type,
     choice_set,
     parking_tariffs,
     parking_fines,
     parking_cost,
     vehicle_attributes,
     parking_congestion,
     parking_capacity,
     explanatory_features,
     utility,
     availability,
     choice,
    ]

edges =\
    [(parking_duration, parking_cost),
     (vehicle_type, parking_cost),
     (parking_tariffs, parking_cost),
     (parking_fines, parking_cost),
     (parking_duration, parking_fines),
     (vehicle_type, choice_set),
     (choice_set, availability),
     (availability, choice),
     (parking_cost, explanatory_features),
     (explanatory_features, utility),
     (utility, choice),
     (vehicle_attributes, explanatory_features),
     (parking_congestion, explanatory_features),
     (parking_capacity, parking_congestion),
    ]

latent_edges = [(parking_congestion, utility)]

parking_causal_model =\
    CausalGraphicalModel(
        nodes=nodes,
        edges=edges,
        latent_edges=latent_edges
    )

parking_causal_model.draw()
# -

# As usual, one immediate question is whether this causal model's testable implications are supported by the author's dataset.
#
# More specific questions arise when one considers the models and policy interventions analyzed in the paper.
# In particular, the authors consider policy interventions that set or alter:
# - parking capacity
# - parking duration
# - parking tariffs
# - parking fines
#
# One is led to immediately wonder how parking duration relates to parking tariffs and vehicle attributes such as the type of vehicle owner (e.g. large corporation vs independent truck owner) and the type of goods being transported (large heavy loads vs small loads).
# Additionally, it might be reasonable to expect that there may be unobserved confounders that lead to loading or unloading delays for all drivers at a particular parking facility, thus affecting both parking congestion and parking duration.
# Finally, it makes sense that at locations with higher parking tariffs, that driver's and workers may have greater incentive to minimize parking duration to avoid higher costs.
#
# Chiara et al., state that they tread parking duration as "exogenous and known by the driver before making a parking choice" (p. 7).
# Given the concerns in the preceding paragraph, the assumption that parking duration is exogeneous is a-priori suspect.
# The causal graph above suggests testing this assumption by attempting to falsify the implicit assumptions of:
# - marginal independence between vehicle attributes and parking duration
# - marginal independence between parking duration and parking tariffs
# - marginal independence between parking duration and parking congestion.
#
# The exogeneity assumption is crucially important, as noted by Chiara et al.
# From a statistical perspective, the authors note on page 14 that,
# > One necessary condition to obtain unbiased estimates of the unknown parameters is the exogeneity of the explana-tory variables.
# Whenever an observable covariate is correlated with unobserved factors contained in the error term (hence its endogeneity), its coefficient estimate will capture not only the effect of the variable itself but also the effect of the correlated unobservedfactors on the utility (Train, 2003).
#
# From a causal perspective, the causal effect of the author's considered policy interventions depends on the assumed causal relationships.
# **If parking duration is partially caused by parking tariffs, then the estimated causal effects of changing parking tariffs may be highly inaccurate because these downstream effects were ignored.**
# Likewise, if parking duration is partially caused by vehicle attributes such as the type of goods being transported, then the considered interventions that limit parking durations to a given time may be unrealistic (i.e., not useful).
# There may be physical limits to how low one's parking duration can be, given the type and quantity of goods being delivered or picked-up. 
