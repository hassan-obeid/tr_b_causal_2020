# -*- coding: utf-8 -*-
"""
Causal graph implied when applying the deconfounder algorithm of Wang and Blei
(2019) to the dataset of Brathwaite and Walker (2018) and the drive-alone
utility in particular.

References
----------
Yixin Wang and David M Blei. The blessings of multiple causes. Journal of the
American Statistical Association, 114(528):1574–1596, 2019.

Brathwaite, Timothy, and Joan L. Walker. "Asymmetric, closed-form,
finite-parameter models of multinomial choice." Journal of choice modelling 29
(2018): 78-112.
"""
import graphviz

DECONFOUNDER_GRAPH = graphviz.Digraph("Deconfounder Example")

# Add all nodes to the graph
# Use square nodes for observed variables and circular nodes for unoobserved
x_star_alias = "X*"
x_star_label = "<Latent Variables<BR/>X<SUP>*</SUP>>"

_time_label = "time"
_cost_label = "cost"
_distance_label = "distance"
_auto_label = "num_autos"
_driver_label = "num_drivers"
_utility_label = "utility"

DECONFOUNDER_GRAPH.node(x_star_alias, x_star_label, shape="ellipse")
DECONFOUNDER_GRAPH.node(_time_label, "Travel Time\nX_1", shape="box")
DECONFOUNDER_GRAPH.node(_cost_label, "Travel Cost\nX_2", shape="box")
DECONFOUNDER_GRAPH.node(_distance_label, "Travel Distance\nX_3", shape="box")
DECONFOUNDER_GRAPH.node(_auto_label, "Number of Automobiles\nX_4", shape="box")
DECONFOUNDER_GRAPH.node(
    _driver_label, "Number of Licensed Drivers\nX_5", shape="box"
)
DECONFOUNDER_GRAPH.node(_utility_label, "Utility\nU", shape="ellipse")

# Create the graphical chain
observed_nodes = [
    _time_label,
    _cost_label,
    _distance_label,
    _auto_label,
    _driver_label,
]

for node in observed_nodes:
    DECONFOUNDER_GRAPH.edge(x_star_alias, node)
    DECONFOUNDER_GRAPH.edge(node, _utility_label)

DECONFOUNDER_GRAPH.edge(x_star_alias, _utility_label)
