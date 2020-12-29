# -*- coding: utf-8 -*-
"""
Causal graph for the selection on observables section of the manuscript.

"""
import graphviz


# Independent Graph
DA_IND_GRAPH = graphviz.Digraph('Drive Alone Independent Graph')
DA_IND_GRAPH.attr(rankdir='TB')

# Define Edges
DA_IND_GRAPH.edge('Total Travel Time', 'Utility (Drive Alone)')
DA_IND_GRAPH.edge('Number of Autos', 'Utility (Drive Alone)')
DA_IND_GRAPH.edge('Number of Licensed Drivers', 'Utility (Drive Alone)')
DA_IND_GRAPH.edge('Total Travel Cost', 'Utility (Drive Alone)')
DA_IND_GRAPH.edge('Total Travel Distance', 'Utility (Drive Alone)', style='filled', color='red')
DA_IND_GRAPH.edge('Proposed Intervention', 'Total Travel Distance', style='filled', color='red')

## Define specific node characteristics
DA_IND_GRAPH.node('Proposed Intervention', style='filled', color='white')
DA_IND_GRAPH.node('Total Travel Distance', style='outlined', color='red')
DA_IND_GRAPH.node('Utility (Drive Alone)', style='outlined', color='red')

## Drive ALone Interacting Graph
DA_INTERACTING_GRAPH = Digraph('Drive Alone Interacting Graph')
DA_INTERACTING_GRAPH.attr(rankdir='TB')

## Edges
DA_INTERACTING_GRAPH.edge('Proposed Intervention', 'Total Travel Distance', style='filled', color='red')
DA_INTERACTING_GRAPH.edge('Total Travel Distance', 'Utility (Drive Alone)', style='filled', color='red')
DA_INTERACTING_GRAPH.edge('Total Travel Distance', 'Total Travel Time', style='filled', color='red')
DA_INTERACTING_GRAPH.edge('Total Travel Distance', 'Total Travel Cost', style='filled', color='red')
DA_INTERACTING_GRAPH.edge('Total Travel Time', 'Utility (Drive Alone)', style='filled', color='red')
DA_INTERACTING_GRAPH.edge('Number of Autos', 'Utility (Drive Alone)')
DA_INTERACTING_GRAPH.edge('Number of Licensed Drivers', 'Utility (Drive Alone)')
DA_INTERACTING_GRAPH.edge('Total Travel Cost', 'Utility (Drive Alone)', style='filled', color='red')

## Define specific node characteristics
DA_INTERACTING_GRAPH.node('Proposed Intervention', style='filled', color='white')
DA_INTERACTING_GRAPH.node('Total Travel Distance', style='outlined', color='red')
DA_INTERACTING_GRAPH.node('Total Travel Time', style='outlined', color='red')
DA_INTERACTING_GRAPH.node('Total Travel Cost', style='outlined', color='red')
DA_INTERACTING_GRAPH.node('Utility (Drive Alone)', style='outlined', color='red')

# Shared Ride 2 Interacting Graph
SHARED_2_INTERACTING_GRAPH = Digraph('Shared Ride 2 Alone Interacting Graph')
SHARED_2_INTERACTING_GRAPH.attr(rankdir='TB')

## Define Edges
SHARED_2_INTERACTING_GRAPH.edge('Number of Kids', 'Utility (Shared Ride 2)')
SHARED_2_INTERACTING_GRAPH.edge('Proposed Intervention', 'Total Travel Distance', style='filled', color='red')
SHARED_2_INTERACTING_GRAPH.edge('Total Travel Distance', 'Utility (Shared Ride 2)', style='filled', color='red')
SHARED_2_INTERACTING_GRAPH.edge('Total Travel Distance', 'Total Travel Time', style='filled', color='red')
SHARED_2_INTERACTING_GRAPH.edge('Total Travel Distance', 'Total Travel Cost', style='filled', color='red')
SHARED_2_INTERACTING_GRAPH.edge('Total Travel Time', 'Utility (Shared Ride 2)', style='filled', color='red')
SHARED_2_INTERACTING_GRAPH.edge('Number of Autos', 'Utility (Shared Ride 2)')
SHARED_2_INTERACTING_GRAPH.edge('Household Size', 'Utility (Shared Ride 2)')
SHARED_2_INTERACTING_GRAPH.edge('Number of Licensed Drivers', 'Utility (Shared Ride 2)')
SHARED_2_INTERACTING_GRAPH.edge('Cross Bay Trip', 'Utility (Shared Ride 2)')
SHARED_2_INTERACTING_GRAPH.edge('Total Travel Cost', 'Utility (Shared Ride 2)', style='filled', color='red')

## Nodes
SHARED_2_INTERACTING_GRAPH.node('Proposed Intervention', style='filled', color='white')
SHARED_2_INTERACTING_GRAPH.node('Total Travel Distance', style='outlined', color='red')
SHARED_2_INTERACTING_GRAPH.node('Total Travel Time', style='outlined', color='red')
SHARED_2_INTERACTING_GRAPH.node('Total Travel Cost', style='outlined', color='red')
SHARED_2_INTERACTING_GRAPH.node('Utility (Shared Ride 2)', style='outlined', color='red')

# Shared Ride 3 Interacting Graph
SHARED_3P_INTERACTING_GRAPH = Digraph('Shared Ride 3+ Interacting Graph')
SHARED_3P_INTERACTING_GRAPH.attr(rankdir='TB')

# Edges
SHARED_3P_INTERACTING_GRAPH.edge('Number of Kids', 'Utility (Shared Ride 3+)')
SHARED_3P_INTERACTING_GRAPH.edge('Proposed Intervention', 'Total Travel Distance', style='filled', color='red')
SHARED_3P_INTERACTING_GRAPH.edge('Total Travel Distance', 'Utility (Shared Ride 3+)', style='filled', color='red')
SHARED_3P_INTERACTING_GRAPH.edge('Total Travel Distance', 'Total Travel Time', style='filled', color='red')
SHARED_3P_INTERACTING_GRAPH.edge('Total Travel Distance', 'Total Travel Cost', style='filled', color='red')
SHARED_3P_INTERACTING_GRAPH.edge('Total Travel Time', 'Utility (Shared Ride 3+)', style='filled', color='red')
SHARED_3P_INTERACTING_GRAPH.edge('Number of Autos', 'Utility (Shared Ride 3+)')
SHARED_3P_INTERACTING_GRAPH.edge('Household Size', 'Utility (Shared Ride 3+)')
SHARED_3P_INTERACTING_GRAPH.edge('Number of Licensed Drivers', 'Utility (Shared Ride 3+)')
SHARED_3P_INTERACTING_GRAPH.edge('Cross Bay Trip', 'Utility (Shared Ride 3+)')
SHARED_3P_INTERACTING_GRAPH.edge('Total Travel Cost', 'Utility (Shared Ride 3+)', style='filled', color='red')

# Nodes
SHARED_3P_INTERACTING_GRAPH.node('Proposed Intervention', style='filled', color='white')
SHARED_3P_INTERACTING_GRAPH.node('Total Travel Distance', style='outlined', color='red')
SHARED_3P_INTERACTING_GRAPH.node('Total Travel Time', style='outlined', color='red')
SHARED_3P_INTERACTING_GRAPH.node('Total Travel Cost', style='outlined', color='red')
SHARED_3P_INTERACTING_GRAPH.node('Utility (Shared Ride 3+)', style='outlined', color='red')
