# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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


# +
## Specify the true causal graph used in the simulation of the choices

causal_graph = CausalGraphicalModel(
    nodes=["X", "Y", "Z"],
    edges=[
        ("X", "Z"),
        ("X", "Y"),
        ("Z", "Y"),
    ],
)

# draw return a graphviz `dot` object, which jupyter can render
figure = causal_graph.draw()

figure.render(
    filename="simple-graph",
)

figure

# +
## Specify the true causal graph used in the simulation of the choices


causal_graph = CausalGraphicalModel(
    nodes=["X", "Y", "Z", "U"],
    edges=[
        ("X", "Z"),
        ("X", "Y"),
        ("Z", "Y"),
        ("U", "Z"),
        ("U", "Y"),
    ],
)

# draw return a graphviz `dot` object, which jupyter can render
figure = causal_graph.draw()

figure.render(
    filename="simple-graph-confounded",
)

figure
# -




