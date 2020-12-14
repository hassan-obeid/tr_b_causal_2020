# -*- coding: utf-8 -*-
"""
Graphical models of the drive alone utility for our choice model, assuming
strong-ignorability (`DRIVE_ALONE_UTILITY`) and under the assumptions of
unobserved confounding (`LATENT_DRIVE_ALONE_UTILITY`) from "The Blessings of
Multiple Causes" (2018) by Wang and Blei.
"""
from causalgraphicalmodels import CausalGraphicalModel

DRIVE_ALONE_UTILITY = CausalGraphicalModel(
    nodes=[
        "Total Travel Distance",
        "Total Travel Time",
        "Total Travel Cost",
        "Number of Autos",
        "Number of Licensed Drivers",
        "Utility (Drive Alone)",
    ],
    edges=[
        ("Total Travel Distance", "Total Travel Time"),
        ("Total Travel Distance", "Total Travel Cost"),
        ("Total Travel Distance", "Utility (Drive Alone)"),
        ("Total Travel Time", "Utility (Drive Alone)"),
        ("Total Travel Cost", "Utility (Drive Alone)"),
        ("Number of Autos", "Utility (Drive Alone)"),
        ("Number of Licensed Drivers", "Utility (Drive Alone)"),
    ],
)

drive_alone_nodes = list(DRIVE_ALONE_UTILITY.dag.nodes)
nodes_for_latent_graph = ["confounder"] + drive_alone_nodes
edges_for_latent_graph = [("confounder", x) for x in drive_alone_nodes] + [
    (x, "Utility (Drive Alone)") for x in drive_alone_nodes[:-1]
]

LATENT_DRIVE_ALONE_UTILITY = CausalGraphicalModel(
    nodes=nodes_for_latent_graph, edges=edges_for_latent_graph
)
