# -*- coding: utf-8 -*-
"""
Causal graph for the random utility model in Ben-Akiva et al. (2002).

References
----------
Ben-Akiva, Moshe, Joan Walker, Adriana T. Bernardino, Dinesh A. Gopinath,
Taka Morikawa, and Amalia Polydoropoulou. "Integration of choice and latent
variable models." Perpetual motion: Travel behaviour research opportunities and
application challenges (2002): 431-470.
"""
import graphviz

RUM_GRAPH = graphviz.Digraph("Random Utility Maximization")

# Add all nodes to the graph
# Use square nodes for observed variables and circular nodes for unoobserved
RUM_GRAPH.node("X", "Explanatory Variables", shape="box")
RUM_GRAPH.node("U", "Utility", shape="ellipse")
RUM_GRAPH.node("C", "Choice", shape="box")

# Create the graphical chain
RUM_GRAPH.edge("X", "U")
RUM_GRAPH.edge("U", "C")
