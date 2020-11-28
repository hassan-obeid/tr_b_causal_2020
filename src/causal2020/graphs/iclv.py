# -*- coding: utf-8 -*-
"""
Causal graph for the integrated choice and latent variable model in Ben-Akiva
et al. (2002).

References
----------
Ben-Akiva, Moshe, Joan Walker, Adriana T. Bernardino, Dinesh A. Gopinath,
Taka Morikawa, and Amalia Polydoropoulou. "Integration of choice and latent
variable models." Perpetual motion: Travel behaviour research opportunities and
application challenges (2002): 431-470.
"""
import graphviz

ICLV_GRAPH = graphviz.Digraph("Integrated Choice and Latent Variable Model")

# Add all nodes to the graph
# Use square nodes for observed variables and circular nodes for unoobserved
x_star_alias = "X*"
x_star_label = "<Latent Variables<BR/>X<SUP>*</SUP>>"

ICLV_GRAPH.node("X", "Explanatory Variables\nX", shape="box")
ICLV_GRAPH.node(x_star_alias, x_star_label, shape="ellipse")
ICLV_GRAPH.node("I", "Indicators\nI", shape="box")
ICLV_GRAPH.node("U", "Utility\nU", shape="ellipse")
ICLV_GRAPH.node("C", "Choice\nC", shape="box")

# Create the graphical chain
ICLV_GRAPH.edge("X", "U")
ICLV_GRAPH.edge("X", x_star_alias)
ICLV_GRAPH.edge(x_star_alias, "I")
ICLV_GRAPH.edge(x_star_alias, "U")
ICLV_GRAPH.edge("U", "C")
