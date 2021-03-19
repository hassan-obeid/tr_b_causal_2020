# -*- coding: utf-8 -*-
"""
Code for the Drive Alone Independent Causal Graph in the
selection on observables section of the manuscript.
"""
import graphviz

# Independent Graph
DA_IND_GRAPH = graphviz.Digraph("Drive Alone Independent Graph")
DA_IND_GRAPH.attr(rankdir="TB")

# Define Edges
DA_IND_GRAPH.edge("Total Travel Time", "Utility (Drive Alone)")
DA_IND_GRAPH.edge("Number of Autos", "Utility (Drive Alone)")
DA_IND_GRAPH.edge("Number of Licensed Drivers", "Utility (Drive Alone)")
DA_IND_GRAPH.edge("Total Travel Cost", "Utility (Drive Alone)")
DA_IND_GRAPH.edge("Cross Bay Trip", "Utility (Drive Alone)")
DA_IND_GRAPH.edge(
    "Total Travel Distance",
    "Utility (Drive Alone)",
    style="filled",
    color="red",
)
DA_IND_GRAPH.edge(
    "Proposed Intervention",
    "Total Travel Distance",
    style="filled",
    color="red",
)

## Define specific node characteristics
DA_IND_GRAPH.node("Proposed Intervention", style="filled", color="white")
DA_IND_GRAPH.node("Total Travel Distance", style="outlined", color="red")
DA_IND_GRAPH.node("Utility (Drive Alone)", style="outlined", color="red")
