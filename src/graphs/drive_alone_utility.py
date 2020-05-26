# -*- coding: utf-8 -*-
"""
A graphical model of the drive alone utility for our choice model.
"""
from causalgraphicalmodels import CausalGraphicalModel

DRIVE_ALONE_UTILITY =\
    CausalGraphicalModel(
        nodes=["Total Travel Distance",
               "Total Travel Time",
               "Total Travel Cost",
               "Number of Autos",
               "Number of Licensed Drivers",
               "Utility (Drive Alone)"],
        edges=[("Total Travel Distance", "Total Travel Time"),
               ("Total Travel Distance", "Total Travel Cost"),
               ("Total Travel Distance", "Utility (Drive Alone)"),
               ("Total Travel Time", "Utility (Drive Alone)"),
               ("Total Travel Cost", "Utility (Drive Alone)"),
               ("Number of Autos", "Utility (Drive Alone)"),
               ("Number of Licensed Drivers", "Utility (Drive Alone)")]
        )
