# -*- coding: utf-8 -*-
"""
Causal graph for the parking choice model in Chiara et al. (2020).

References
----------
Chiara, Giacomo Dalla, Lynette Cheah, Carlos Lima Azevedo, and Moshe E. Ben-Akiva.
"A Policy-Sensitive Model of Parking Choice for Commercial Vehicles in Urban Areas."
Transportation Science (2020).
"""
from causalgraphicalmodels import CausalGraphicalModel

# Note that string representations stored
# as objects so I can later easily change the
# string representations in one place and update
# the representation everywhere
parking_duration = "parking_duration\n(d_n)"
vehicle_type = "vehicle_type\n(v_n)"
choice_set = "choice_set\n(C_n)"
parking_tariffs = "parking_tariffs"
parking_fines = "parking_fines"
parking_cost = "parking_cost\n(C_ni)"
vehicle_attributes = "vehicle_attributes\n(Z_n)"
parking_congestion = "parking_congestion\n(q_it)"
parking_capacity = "parking_capacity\n(N_i)"
previous_arrivals = "previous_arrivals\n(arr_t)"
explanatory_features = "explanatory_features\n(X_nit)"
unobserved_confounders = "unobserved_confounders"
utility = "utility\n(U_nit)"
choice = "choice\n(Y_nit)"

nodes = [
    parking_duration,
    vehicle_type,
    choice_set,
    parking_tariffs,
    parking_fines,
    parking_cost,
    vehicle_attributes,
    parking_congestion,
    parking_capacity,
    previous_arrivals,
    explanatory_features,
    utility,
    choice,
]

edges = [
    (parking_duration, parking_cost),
    (vehicle_type, parking_fines),
    (parking_tariffs, parking_cost),
    (parking_fines, parking_cost),
    (parking_duration, parking_fines),
    (vehicle_type, choice_set),
    (choice_set, choice),
    (parking_cost, explanatory_features),
    (explanatory_features, utility),
    (utility, choice),
    (vehicle_attributes, explanatory_features),
    (parking_congestion, explanatory_features),
    (parking_capacity, parking_congestion),
    (previous_arrivals, parking_congestion),
]

latent_edges = [(parking_congestion, utility)]

PARKING_CAUSAL_MODEL = CausalGraphicalModel(
    nodes=nodes, edges=edges, latent_edges=latent_edges
)
