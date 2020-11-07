"""
Graphical Models of all the utility functions based on the MNL model from
"Asymmetric, Closed-Form, Finite-Parameter Models of Multinomial Choice"
By Brathwaite & Walker (2016).
"""

from causalgraphicalmodels import CausalGraphicalModel

# Definition of causal graphs nodes and edges

# Independent
NODES_IND = ["Total Travel Distance",
             "Total Travel Time",
             "Total Travel Cost",
             "Number of Licensed Drivers",
             "Number of Autos",
             "Utility (Drive Alone)"]

EDGES_IND = [("Total Travel Distance", "Utility (Drive Alone)"),
             ("Total Travel Time", "Utility (Drive Alone)"),
             ("Total Travel Cost", "Utility (Drive Alone)"),
             ("Number of Licensed Drivers", "Utility (Drive Alone)"),
             ("Number of Autos", "Utility (Drive Alone)")]

IND_UTILITY = CausalGraphicalModel(NODES_IND, EDGES_IND)

# Drive Alone
NODES_DA = ["Total Travel Distance",
            "Total Travel Time",
            "Total Travel Cost",
            "Number of Autos",
            "Number of Licensed Drivers",
            "Utility (Drive Alone)"]

EDGES_DA = [("Total Travel Distance", "Total Travel Time"),
            ("Total Travel Distance", "Total Travel Cost"),
            ("Total Travel Distance", "Utility (Drive Alone)"),
            ("Total Travel Time", "Utility (Drive Alone)"),
            ("Total Travel Cost", "Utility (Drive Alone)"),
            ("Number of Autos", "Utility (Drive Alone)"),
            ("Number of Licensed Drivers", "Utility (Drive Alone)")]

DA_UTILITY = CausalGraphicalModel(NODES_DA, EDGES_DA)

# Shared-2
NODES_SHARED_2 = ["Total Travel Time",
                  "Total Travel Distance",
                  "Total Travel Cost",
                  "Cross Bay Trip",
                  "Number of Autos",
                  "Number of Licensed Drivers",
                  "Household Size",
                  "Number of Kids",
                  "Utility (Shared Ride 2)"]

EDGES_SHARED_2 = [("Total Travel Distance", "Total Travel Time"),
                  ("Total Travel Distance", "Total Travel Cost"),
                  ("Total Travel Distance", "Utility (Shared Ride 2)"),
                  ("Total Travel Time", "Utility (Shared Ride 2)"),
                  ("Number of Autos", "Utility (Shared Ride 2)"),
                  ("Number of Licensed Drivers", "Utility (Shared Ride 2)"),
                  ("Total Travel Cost", "Utility (Shared Ride 2)"),
                  ("Household Size", "Utility (Shared Ride 2)"),
                  ("Cross Bay Trip", "Utility (Shared Ride 2)"),
                  ("Number of Kids", "Utility (Shared Ride 2)")]

SHARED_2_UTILITY = CausalGraphicalModel(NODES_SHARED_2, EDGES_SHARED_2)

# Shared-3+
NODES_SHARED_3P = ["Total Travel Time",
                   "Total Travel Distance",
                   "Total Travel Cost",
                   "Cross Bay Trip",
                   "Number of Autos",
                   "Number of Licensed Drivers",
                   "Household Size",
                   "Number of Kids",
                   "Utility (Shared Ride 3+)"]

EDGES_SHARED_3P = [("Total Travel Distance", "Total Travel Time"),
                   ("Total Travel Distance", "Total Travel Cost"),
                   ("Total Travel Distance", "Utility (Shared Ride 3+)"),
                   ("Total Travel Time", "Utility (Shared Ride 3+)"),
                   ("Number of Autos", "Utility (Shared Ride 3+)"),
                   ("Number of Licensed Drivers", "Utility (Shared Ride 3+)"),
                   ("Total Travel Cost", "Utility (Shared Ride 3+)"),
                   ("Household Size", "Utility (Shared Ride 3+)"),
                   ("Cross Bay Trip", "Utility (Shared Ride 3+)"),
                   ("Number of Kids", "Utility (Shared Ride 3+)")]

SHARED_3P_UTILITY = CausalGraphicalModel(NODES_SHARED_3P, EDGES_SHARED_3P)

# Walk-Transit-Walk
NODES_WTW = ["Total Travel Time",
             "Total Travel Cost",
             "Utility (WTW)"]

EDGES_WTW = [("Total Travel Time", "Total Travel Cost"),
             ("Total Travel Time", "Utility (WTW)"),
             ("Total Travel Cost", "Utility (WTW)")]

WTW_UTILITY = CausalGraphicalModel(NODES_WTW, EDGES_WTW)

# Drive-Transit-Walk
NODES_DTW = ["Total Travel Time",
             "Total Travel Cost",
             "Utility (DTW)"]

EDGES_DTW = [("Total Travel Time", "Total Travel Cost"),
             ("Total Travel Time", "Utility (DTW)"),
             ("Total Travel Cost", "Utility (DTW)")]

DTW_UTILITY = CausalGraphicalModel(NODES_DTW, EDGES_DTW)

# Walk-Transit-Drive
NODES_WTD = ["Total Travel Time",
             "Total Travel Cost",
             "Utility (WTD)"]

EDGES_WTD = [("Total Travel Time", "Total Travel Cost"),
             ("Total Travel Time", "Utility (WTD)"),
             ("Total Travel Cost", "Utility (WTD)")]

WTD_UTILITY = CausalGraphicalModel(NODES_WTD, EDGES_WTD)

# Walk
NODES_WALK = ["Total Travel Distance",
              "Utility (Walk)"]

EDGES_WALK = [("Total Travel Distance", "Utility (Walk)")]

WALK_UTILITY = CausalGraphicalModel(NODES_WALK, EDGES_WALK)

# Bike
NODES_BIKE = ["Total Travel Distance",
              "Utility (Bike)"]
EDGES_BIKE = [("Total Travel Distance", "Utility (Bike)")]

BIKE_UTILITY = CausalGraphicalModel(NODES_BIKE, EDGES_BIKE)
