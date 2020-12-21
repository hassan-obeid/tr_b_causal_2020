# -*- coding: utf-8 -*-
"""
Stores all causal graphs for the utility equations in Section 2 of the article.
"""
import causal2020.observables.graphs as graphs
import click
from causal2020 import utils


def main_func(supplier=graphs, writer=utils) -> bool:
    sob_graphs = {
        "Independent_graph": supplier.IND_UTILITY,
        "DA_interacting_graph": supplier.DA_UTILITY,
        "SR2_interacting_graph": supplier.SHARED_2_UTILITY,
        "SR3_interacting_graph": supplier.SHARED_3P_UTILITY,
        "WTW_interacting_graph": supplier.WTW_UTILITY,
        "DTW_interacting_graph": supplier.DTW_UTILITY,
        "WTD_interacting_graph": supplier.WTD_UTILITY,
        "BIKE_interacting_graph": supplier.BIKE_UTILITY,
        "WALK_interacting_graph": supplier.WALK_UTILITY,
    }

    # Write the image of each graph to file
    for output_name in sob_graphs:
        writer.create_graph_image(
            graph=sob_graphs[output_name],
            output_name=output_name,
            output_type="pdf",
        )

    return True


@click.command()
def main() -> bool:
    main_func()


if __name__ == "__main__":
    main()
