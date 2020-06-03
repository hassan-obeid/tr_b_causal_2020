# -*- coding: utf-8 -*-
"""
A graphical model to demonstrate conditional independence. Shows a graph that
assumes travel cost is conditionally independent of travel time, conditional on
travel distance.
"""
from causalgraphicalmodels import CausalGraphicalModel

TRAVEL_TIME = 'Travel Time'
TRAVEL_COST = 'Travel Cost'
TRAVEL_DISTANCE = 'Travel Distance'


EXAMPLE_GRAPH =\
    CausalGraphicalModel(
        nodes=[TRAVEL_TIME, TRAVEL_COST, TRAVEL_DISTANCE],
        edges=[(TRAVEL_DISTANCE, TRAVEL_TIME), (TRAVEL_DISTANCE, TRAVEL_COST)]
    )
