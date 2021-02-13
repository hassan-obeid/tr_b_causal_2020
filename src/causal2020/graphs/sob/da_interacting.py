# -*- coding: utf-8 -*-
"""
Code for the Drive Alone Interacting Causal Graph in the
selection on observables section of the manuscript.
"""
import graphviz

## Drive ALone Interacting Graph
DA_INTERACTING_GRAPH = graphviz.Digraph("Drive Alone Interacting Graph")
DA_INTERACTING_GRAPH.attr(rankdir="TB")

## Edges
DA_INTERACTING_GRAPH.edge(
    "Proposed Intervention",
    "Total Travel Distance",
    style="filled",
    color="red",
)
DA_INTERACTING_GRAPH.edge(
    "Total Travel Distance",
    "Utility (Drive Alone)",
    style="filled",
    color="red",
)
DA_INTERACTING_GRAPH.edge(
    "Total Travel Distance", "Total Travel Time", style="filled", color="red"
)
DA_INTERACTING_GRAPH.edge(
    "Total Travel Distance", "Total Travel Cost", style="filled", color="red"
)
DA_INTERACTING_GRAPH.edge(
    "Total Travel Time", "Utility (Drive Alone)", style="filled", color="red"
)
DA_INTERACTING_GRAPH.edge("Number of Autos", "Utility (Drive Alone)")
DA_INTERACTING_GRAPH.edge(
    "Number of Licensed Drivers", "Utility (Drive Alone)"
)
DA_INTERACTING_GRAPH.edge("Cross Bay Trip", "Utility (Drive Alone)")
DA_INTERACTING_GRAPH.edge(
    "Total Travel Cost", "Utility (Drive Alone)", style="filled", color="red"
)

## Define specific node characteristics
DA_INTERACTING_GRAPH.node(
    "Proposed Intervention", style="filled", color="white"
)
DA_INTERACTING_GRAPH.node(
    "Total Travel Distance", style="outlined", color="red"
)
DA_INTERACTING_GRAPH.node("Total Travel Time", style="outlined", color="red")
DA_INTERACTING_GRAPH.node("Total Travel Cost", style="outlined", color="red")
DA_INTERACTING_GRAPH.node(
    "Utility (Drive Alone)", style="outlined", color="red"
)
