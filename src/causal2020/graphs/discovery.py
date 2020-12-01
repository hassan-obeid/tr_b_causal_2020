# -*- coding: utf-8 -*-
"""
Causal graph resulting from the PC algorithm (Glymour et al., 2001) and the
data from Brathwaite and Walker (2018).

References
----------
Brathwaite, Timothy, and Joan L. Walker. "Asymmetric, closed-form,
finite-parameter models of multinomial choice." Journal of choice modelling 29
(2018): 78-112.

Clark Glymour, Richard Scheines, and Peter Spirtes. Causation, prediction, and
search. MIT Press, 2001.
"""
import graphviz

TIME_COLUMN = "Travel Time"
COST_COLUMN = "Travel Cost"
DISTANCE_COLUMN = "Travel Distance"
LICENSE_COLUMN = "Number of Licensed Drivers"
NUM_AUTOS_COLUMN = "Number of Automobiles"

DISCOVERY_GRAPH = graphviz.Digraph("DISCOVERY")

# Add all nodes to the graph
DISCOVERY_GRAPH.node("T", TIME_COLUMN)
DISCOVERY_GRAPH.node("C", COST_COLUMN)
DISCOVERY_GRAPH.node("D", DISTANCE_COLUMN)
DISCOVERY_GRAPH.node("L", LICENSE_COLUMN)
DISCOVERY_GRAPH.node("A", NUM_AUTOS_COLUMN)

# Add edges to the graph
DISCOVERY_GRAPH.edge("L", "A")
DISCOVERY_GRAPH.edge("D", "A")
DISCOVERY_GRAPH.edge("D", "T")
DISCOVERY_GRAPH.edge("D", "C")
DISCOVERY_GRAPH.edge("C", "T", dir="none")

# Display the graph
# utils.create_graph_image(
#     step_9_graph, output_name=PLOT_TITLE, output_type="pdf"
# )

DISCOVERY_GRAPH.graph_attr.update(size="10,6")
DISCOVERY_GRAPH.node_attr.update(shape="box")
DISCOVERY_GRAPH
